from flask import Blueprint, request, jsonify
import jpype
import pandas as pd
import os
import numpy as np
from collections import Counter
import random

bp = Blueprint('main', __name__)

# 启动JVM并添加ARX库路径
if not jpype.isJVMStarted():
    jar_path = os.path.abspath("arx-3.9.1-osx-64.jar")
    print(f"JAR file path: {jar_path}")
    jpype.startJVM(classpath=[jar_path])
    print("JVM started:", jpype.isJVMStarted())
    print("JVM started with classpath:", jpype.java.lang.System.getProperty("java.class.path"))

# 导入ARX库的Java类
try:
    Data = jpype.JClass("org.deidentifier.arx.Data")
    ARXAnonymizer = jpype.JClass("org.deidentifier.arx.ARXAnonymizer")
    ARXConfiguration = jpype.JClass("org.deidentifier.arx.ARXConfiguration")
    AttributeType = jpype.JClass("org.deidentifier.arx.AttributeType")
    KAnonymity = jpype.JClass("org.deidentifier.arx.criteria.KAnonymity")
    LDiversity = jpype.JClass("org.deidentifier.arx.criteria.LDiversity")
    TCloseness = jpype.JClass("org.deidentifier.arx.criteria.TCloseness")
except Exception as e:
    print("Failed to load ARX classes:", e)

def create_numeric_hierarchy(values, num_levels=3):
    min_val, max_val = min(values), max(values)
    step = (max_val - min_val) / num_levels
    hierarchy = []
    for i in range(num_levels):
        bins = np.arange(min_val, max_val + step, step)
        bin_labels = [f"{int(b)}-{int(b+step-1)}" for b in bins[:-1]]
        hierarchy.insert(0, bin_labels)
    hierarchy.append([str(int(v)) for v in sorted(set(values))])
    return hierarchy

def create_categorical_hierarchy(values):
    unique_values = sorted(set(values))
    hierarchy = [["*"], unique_values]
    return hierarchy

def load_data(file_path, quasi_identifiers):
    df = pd.read_csv(file_path)

    data = Data.create()
    columns = list(df.columns)
    data.add(jpype.JArray(jpype.JString)(columns))
    for index, row in df.iterrows():
        data.add(jpype.JArray(jpype.JString)(list(row.astype(str))))

    # 导入ARX的必需类
    HierarchyBuilderRedactionBased = jpype.JClass("org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased")
    HierarchyBuilderIntervalBased = jpype.JClass("org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased")
    DataType = jpype.JClass("org.deidentifier.arx.DataType")

    for column in quasi_identifiers:
        values = df[column].dropna().values
        if np.issubdtype(values.dtype, np.number):
            # 使用 HierarchyBuilderIntervalBased 创建数值型数据的泛化层次结构
            builder = HierarchyBuilderIntervalBased.create(DataType.DECIMAL)
            data.getDefinition().setAttributeType(column, builder)
        else:
            # 使用 HierarchyBuilderRedactionBased.create(mask_char) 来创建基于字符的层次结构
            mask_char = '*'  # 使用 '*' 作为泛化字符
            builder = HierarchyBuilderRedactionBased.create(mask_char)
            data.getDefinition().setAttributeType(column, builder)

    return data


def apply_k_anonymity(data, k_value):
    config = ARXConfiguration.create()
    config.addPrivacyModel(KAnonymity(k_value))
    anonymizer = ARXAnonymizer()
    result = anonymizer.anonymize(data, config)
    return result

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd

# Activate automatic conversion between pandas dataframes and R data.frames
pandas2ri.activate()

def apply_l_diversity_r(dataframe, quasi_identifiers, sensitive_column, l_value):
    try:
        # Convert the Pandas DataFrame to an R DataFrame
        r_dataframe = pandas2ri.py2rpy(dataframe)
        
        # Assign the R DataFrame to the R global environment
        robjects.globalenv['r_dataframe'] = r_dataframe

        # Convert the sensitive column to a factor using R's within() function
        robjects.r(f"""
            r_dataframe <- within(r_dataframe, {{
                {sensitive_column} <- as.factor({sensitive_column})
            }})
        """)
        
        # Import the sdcMicro package
        sdcmicro = importr('sdcMicro')

        # Define quasi-identifiers as an R vector
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # Create an SdcMicro object with the data
        sdc_obj = sdcmicro.createSdcObj(dat=robjects.globalenv['r_dataframe'], keyVars=quasi_identifiers_r)

        # Get the index of the sensitive column (R indexing starts at 1)
        sensitive_column_index = dataframe.columns.get_loc(sensitive_column) + 1

        # Apply l-diversity using the ldiversity function
        ldiversity_result = sdcmicro.ldiversity(sdc_obj, ldiv_index=robjects.IntVector([sensitive_column_index]), l_recurs_c=l_value)

        # Check for the 'manipKeyVars' slot in the result
        if 'manipKeyVars' in robjects.r.slotNames(ldiversity_result):
            anonymized_data = robjects.r.slot(ldiversity_result, 'manipKeyVars')
            anonymized_df = pandas2ri.rpy2py(anonymized_data)
            return anonymized_df
        else:
            raise ValueError("Expected slot 'manipKeyVars' not found in ldiversity_result")

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise


def apply_t_closeness(data, attribute, t_value):
    config = ARXConfiguration.create()
    print(f"Applying t-closeness with attribute: {attribute}, t_value: {t_value}")
    try:
        t_closeness_model = TCloseness(attribute, float(t_value))  # 确保t_value是浮点数
        config.addPrivacyModel(t_closeness_model)
    except Exception as e:
        print(f"Error while configuring TCloseness: {e}")
        raise

    anonymizer = ARXAnonymizer()
    result = anonymizer.anonymize(data, config)
    return result


def convert_to_dataframe(result):
    # 从ARXResult中获取DataHandle
    handle = result.getOutput()

    # 检查匿名化结果的状态
    if handle is None:
        raise ValueError("Anonymization failed to produce output.")

    # 获取匿名化后的数据
    anon_data = []
    iterator = handle.iterator()
    while iterator.hasNext():
        row = iterator.next()
        row_data = [str(cell) for cell in row]
        anon_data.append(row_data)

    # 获取列名
    columns = [handle.getAttributeName(i) for i in range(handle.getNumColumns())]

    # 创建 DataFrame
    anon_df = pd.DataFrame(anon_data, columns=columns)
    return anon_df

@bp.route('/anonymize', methods=['POST'])
def anonymize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    privacy_model = request.form.get('privacy_model', 'k-anonymity')  # 默认使用 k-anonymity
    k_value = int(request.form.get('k', 2))
    l_value = int(request.form.get('l', 2))  # 确保 l_value 是整数
    t_value = float(request.form.get('t', 0.2))
    attribute = request.form.get('attribute', 'gender')  # 默认属性为 gender
    quasi_identifiers = request.form.get('quasi_identifiers', 'gender,Age,zipcode').split(',')  # 默认准标识符
    sensitive_column = request.form.get('sensitive_column', 'Disease')  # 默认敏感属性为 Disease
    file_path = os.path.join(bp.root_path, 'uploads', file.filename)
    file.save(file_path)

    dataPd = pd.read_csv(file_path)
# 检查列是否存在
    for qi in quasi_identifiers:
        if qi not in dataPd.columns:
            return jsonify({'error': f"Column '{qi}' does not exist in the file."}), 400

    # 打印数据类型和前几行数据，进行调试
    print(dataPd.dtypes)
    print(dataPd.head())
    try:
        if privacy_model == 'k-anonymity':
            # 使用现有的 apply_k_anonymity 函数
            arx_data = load_data(file_path, quasi_identifiers)
            result = apply_k_anonymity(arx_data, k_value)
            anon_data = convert_to_dataframe(result)
            return anon_data.to_json(orient='records')
        
        elif privacy_model == 'l-diversity':
            # 使用新的 l-diversity 处理函数
            resultPd = apply_l_diversity_r(dataPd, quasi_identifiers, sensitive_column, l_value)
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        # 打印并返回详细的错误信息
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
# @bp.route('/api/hello', methods=['GET'])
# def hello():
#     return jsonify(message="Hello from Flask!")
