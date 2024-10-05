from flask import Blueprint, request, jsonify
import pandas as pd
import os
import numpy as np  # 添加了这行代码以导入 numpy 库
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

bp = Blueprint('main', __name__)

# Activate automatic conversion between pandas dataframes and R data.frames
pandas2ri.activate()

# Load the necessary R package
sdcmicro = importr('sdcMicro')

def create_generalization_hierarchy(dataframe, quasi_identifiers):
    hierarchies = {}
    for qi in quasi_identifiers:
        unique_values = dataframe[qi].unique()
        if np.issubdtype(dataframe[qi].dtype, np.number):
            min_val, max_val = dataframe[qi].min(), dataframe[qi].max()
            bins = np.linspace(min_val, max_val, num=min(len(unique_values), 4))
            labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
            hierarchy = {}
            for val in unique_values:
                for i in range(len(bins)-1):
                    if bins[i] <= val < bins[i+1]:
                        hierarchy[val] = labels[i]
                        break
                else:
                    hierarchy[val] = labels[-1]
        else:
            hierarchy = {val: "通用类别" for val in unique_values}
        
        hierarchies[qi] = hierarchy

    return hierarchies

def apply_generalization(dataframe, hierarchies):
    generalized_df = dataframe.copy()
    for qi, hierarchy in hierarchies.items():
        generalized_df[qi] = generalized_df[qi].map(hierarchy)
    return generalized_df

def apply_k_anonymity_r(dataframe, quasi_identifiers, k_value):
    try:
        # 创建泛化层次结构
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers)
        
        # 应用泛化
        generalized_df = apply_generalization(dataframe, hierarchies)

        # 转换为R的数据框
        r_dataframe = pandas2ri.py2rpy(generalized_df)

        # 定义准标识符
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 创建SdcMicro对象
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)

        # 应用k-anonymity
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        return anonymized_df

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise

def apply_l_diversity_r(dataframe, quasi_identifiers, sensitive_column, l_value):
    try:
        # 创建泛化层次结构
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers)
        
        # 应用泛化
        generalized_df = apply_generalization(dataframe, hierarchies)

        # 转换为R的数据框
        r_dataframe = pandas2ri.py2rpy(generalized_df)

        # 定义准标识符
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 创建SdcMicro对象
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)

        # 应用l-diversity
        sensitive_column_index = dataframe.columns.get_loc(sensitive_column) + 1
        ldiversity_result = sdcmicro.ldiversity(sdc_obj, ldiv_index=robjects.IntVector([sensitive_column_index]), l_recurs_c=l_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](ldiversity_result)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        return anonymized_df

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise

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
    quasi_identifiers = request.form.get('quasi_identifiers', 'Gender,Age,Zipcode').split(',')  # 默认准标识符
    sensitive_column = request.form.get('sensitive_column', 'Disease')  # 默认敏感属性为 Disease
    file_path = os.path.join(bp.root_path, 'uploads', file.filename)
    file.save(file_path)

    dataPd = pd.read_csv(file_path)

    # 检查列是否存在
    for qi in quasi_identifiers:
        if qi not in dataPd.columns:
            return jsonify({'error': f"Column '{qi}' does not exist in the file."}), 400

    try:
        if privacy_model == 'k-anonymity':
            resultPd = apply_k_anonymity_r(dataPd, quasi_identifiers, k_value)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'l-diversity':
            resultPd = apply_l_diversity_r(dataPd, quasi_identifiers, sensitive_column, l_value)
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
