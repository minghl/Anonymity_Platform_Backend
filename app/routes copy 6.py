from flask import Blueprint, request, jsonify
import pandas as pd
import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from collections import Counter

bp = Blueprint('main', __name__)

# Activate automatic conversion between pandas dataframes and R data.frames
pandas2ri.activate()

# Load the necessary R package
sdcmicro = importr('sdcMicro')

def create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules):
    hierarchies = {}
    for qi in quasi_identifiers:
        unique_values = dataframe[qi].unique()

        if qi in hierarchy_rules:
            rule = hierarchy_rules[qi].lower()
            if rule == 'ordering':
                min_val, max_val = dataframe[qi].min(), dataframe[qi].max()
                bins = np.linspace(min_val, max_val, num=min(len(unique_values), 4))
                labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
                hierarchy = {}
                for val in unique_values:
                    if pd.isna(val):
                        continue
                    bin_index = np.digitize(val, bins) - 1
                    if bin_index >= len(labels):
                        bin_index = len(labels) - 1
                    hierarchy[val] = labels[bin_index]

                # 将空值回填为对应的分层区间
                hierarchy[np.nan] = labels[0]  # 默认将空值分配到第一个区间

            elif rule == 'masking':
                hierarchy = {val: "Masked" for val in unique_values}

            elif rule == 'dates':
                date_range = pd.to_datetime(unique_values)
                
                date_mean = date_range.mean()
                date_std = date_range.std()
                lower_bound = date_mean - 3 * date_std
                upper_bound = date_mean + 3 * date_std
                filtered_dates = date_range[(date_range >= lower_bound) & (date_range <= upper_bound)]
                
                min_date, max_date = filtered_dates.min(), filtered_dates.max()
                time_span = (max_date - min_date).days
                
                if time_span <= 365:
                    num_bins = 2
                elif time_span <= 365 * 5:
                    num_bins = 4
                elif time_span <= 365 * 10:
                    num_bins = 6
                else:
                    num_bins = 8

                bins = pd.date_range(start=min_date, end=max_date, periods=num_bins + 1).astype(np.int64) // 10**9
                labels = [f"{pd.to_datetime(bins[i], unit='s').strftime('%Y-%m-%d')} - "
                          f"{pd.to_datetime(bins[i+1], unit='s').strftime('%Y-%m-%d')}" for i in range(len(bins)-1)]
                
                hierarchy = {}
                for val in unique_values:
                    if pd.isna(val):
                        continue
                    date_val = pd.to_datetime(val).value // 10**9
                    bin_index = np.digitize(date_val, bins) - 1
                    if bin_index >= len(labels):
                        bin_index = len(labels) - 1
                    hierarchy[val] = labels[bin_index]

                # 回填空值为最接近的分层区间
                hierarchy[np.nan] = labels[0]  # 这里默认将空值分配到第一个区间

        else:
            if np.issubdtype(dataframe[qi].dtype, np.number):
                min_val, max_val = dataframe[qi].min(), dataframe[qi].max()
                bins = np.linspace(min_val, max_val, num=min(len(unique_values), 4))
                labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
                hierarchy = {}
                for val in unique_values:
                    if pd.isna(val):
                        continue
                    bin_index = np.digitize(val, bins) - 1
                    if bin_index >= len(labels):
                        bin_index = len(labels) - 1
                    hierarchy[val] = labels[bin_index]

                # 回填空值为最接近的分层区间
                hierarchy[np.nan] = labels[0]

            else:
                hierarchy = {val: "General Category" for val in unique_values}

        hierarchies[qi] = hierarchy

    return hierarchies

def apply_generalization(dataframe, hierarchies):
    generalized_df = dataframe.copy()
    for qi, hierarchy in hierarchies.items():
        generalized_df[qi] = generalized_df[qi].map(hierarchy)
    
    print("Generalized DataFrame (after applying hierarchies):")
    print(generalized_df.head())
    return generalized_df

def check_m_diversity(group, sensitive_column, m_value):
    sensitive_values = group[sensitive_column].values
    value_counts = Counter(sensitive_values)
    return len(value_counts) >= m_value

# 修改的 apply_km_anonymity_r 函数
def apply_km_anonymity_r(dataframe, quasi_identifiers, sensitive_column, k_value, m_value, hierarchy_rules):
    try:
        # 第一步：生成通用化后的数据
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        print(f"Generalized DataFrame before R processing (k={k_value}):")
        print(generalized_df.head())

        # 转换为R的数据格式
        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 使用sdcMicro进行k-匿名性处理
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        print(f"Anonymized DataFrame from R (k={k_value}):")
        print(anonymized_df.head())

        # 第二步：检查 m-多样性
        # 我们按准标识符进行分组，并检查每个组的敏感属性是否满足 m-多样性
        km_anonymized_groups = []
        for _, group in anonymized_df.groupby(quasi_identifiers):
            if len(group) >= k_value and check_m_diversity(group, sensitive_column, m_value):
                km_anonymized_groups.append(group)
            else:
                # 如果不满足m-多样性，可以应用进一步的泛化或抑制处理
                print(f"Group does not meet m-diversity (m={m_value}), additional suppression required.")
        
        # 将满足km-匿名性的组重新合并为最终结果
        km_anonymized_df = pd.concat(km_anonymized_groups)

        print(f"KM-Anonymized DataFrame (k={k_value}, m={m_value}):")
        print(km_anonymized_df.head())

        return km_anonymized_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def apply_k_anonymity_r(dataframe, quasi_identifiers, k_value, hierarchy_rules):
    try:
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        print(f"Generalized DataFrame before R processing (k={k_value}):")
        print(generalized_df.head())

        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        print(f"Anonymized DataFrame from R (k={k_value}):")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise


def apply_t_closeness(dataframe, quasi_identifiers, sensitive_column, t_value, hierarchy_rules):
    try:
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        global_distribution = generalized_df[sensitive_column].value_counts(normalize=True).to_dict()
        groups = generalized_df.groupby(quasi_identifiers)

        for name, group in groups:
            group_distribution = group[sensitive_column].value_counts(normalize=True).to_dict()
            kl_divergence = sum(group_distribution.get(k, 0) * np.log(group_distribution.get(k, 1e-10) / global_distribution.get(k, 1e-10)) for k in global_distribution)

            if kl_divergence > t_value:
                print(f"T-closeness violation in group {name}: KL divergence is {kl_divergence}, which exceeds threshold {t_value}")
                # Additional generalization or handling can be done here

        return generalized_df

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise

def apply_l_diversity_r(dataframe, quasi_identifiers, sensitive_column, l_value, hierarchy_rules):
    try:
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)

        sensitive_column_index = dataframe.columns.get_loc(sensitive_column) + 1
        ldiversity_result = sdcmicro.ldiversity(sdc_obj, ldiv_index=robjects.IntVector([sensitive_column_index]), l_recurs_c=l_value)

        anonymized_data = robjects.r['extractManipData'](ldiversity_result)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        return anonymized_df

    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise
from diffprivlib.tools import mean
import numpy as np
import pandas as pd

def apply_differential_privacy(dataframe, epsilon, quasi_identifiers, hierarchy_rules):
    try:
        # 创建一个新的 DataFrame 用于存储差分隐私处理后的数据
        anonymized_df = dataframe.copy()
        
        for qi in quasi_identifiers:
            if qi in dataframe.columns:
                # 检查数据类型
                rule = hierarchy_rules.get(qi, None)
                
                if rule == "dates":
                    # 日期类型：将日期转换为自某个固定日期以来的天数，这里使用 1970-01-01 也可以使用更早的日期
                    days_since_epoch = (pd.to_datetime(dataframe[qi], errors='coerce') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                    
                    # 添加噪声
                    noise = np.random.laplace(0, 1/epsilon, size=days_since_epoch.shape)
                    noisy_days = days_since_epoch + noise
                    
                    # 将时间戳转换回日期，只保留年月日部分
                    anonymized_df[qi] = pd.to_datetime(noisy_days, origin='1970-01-01', unit='D').dt.date
                    # 转换为字符串格式
                    anonymized_df[qi] = anonymized_df[qi].astype(str)
                elif rule == "ordering" and np.issubdtype(dataframe[qi].dtype, np.number):
                    # 对数值型数据应用差分隐私
                    column_data = pd.to_numeric(dataframe[qi], errors='coerce')
                    column_data = column_data.dropna()  # 删除空值
                    if column_data.empty:
                        print(f"Column '{qi}' is empty after conversion to numeric. Skipping this column.")
                        continue
                    
                    # 添加噪声
                    noise = np.random.laplace(0, 1/epsilon, size=column_data.shape)
                    anonymized_df[qi] = column_data + noise

                elif rule == "masking":
                    # 对标记为 "masking" 的列进行掩码处理
                    anonymized_df[qi] = "Masked"
                    
                else:
                    # 非数值类型和日期类型不处理
                    print(f"Column '{qi}' does not match any known rule. Skipping this column.")
        
        print(f"Differentially Private DataFrame (epsilon={epsilon}):")
        print(anonymized_df.head())

        return anonymized_df
    except Exception as e:
        import traceback
        print("Full traceback of the error:")
        traceback.print_exc()
        raise
import json

def read_file(file_path):
    # 获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lower()

    # 判断是 CSV 还是 TSV
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .tsv file.")



@bp.route('/anonymize', methods=['POST'])
def anonymize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    privacy_model = request.form.get('privacy_model', 'k-anonymity')
    k_value = request.form.get('k', None)
    if k_value:
        k_value = int(k_value)
    
    l_value = request.form.get('l', None)
    if l_value:
        l_value = int(l_value)
    
    t_value = request.form.get('t', None)
    if t_value:
        t_value = float(t_value)
    
    m_value = request.form.get('m', None)
    if m_value:
        m_value = float(m_value)

    epsilon = request.form.get('e', None)
    if epsilon:
        epsilon = float(epsilon)


    quasi_identifiers = request.form.get('quasi_identifiers', 'Gender,Age,Zipcode').split(',')
    sensitive_column = request.form.get('sensitive_column', 'Disease')
    
    # 解析 hierarchy_rules JSON 字符串
    hierarchy_rules = json.loads(request.form.get('hierarchy_rules', '{}'))
    
    file_path = os.path.join(bp.root_path, 'uploads', file.filename)

    file.save(file_path)
    print(file_path)
    dataPd = read_file(file_path)

    for qi in quasi_identifiers:
        if qi not in dataPd.columns:
            return jsonify({'error': f"Column '{qi}' does not exist in the file."}), 400

    try:
        if privacy_model == 'k-anonymity':
            resultPd = apply_k_anonymity_r(dataPd, quasi_identifiers, k_value, hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'l-diversity':
            resultPd = apply_l_diversity_r(dataPd, quasi_identifiers, sensitive_column, l_value, hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 't-closeness':
            resultPd = apply_t_closeness(dataPd, quasi_identifiers, sensitive_column, t_value, hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'km-anonymity':
            print(f"Hierarchy rules: {hierarchy_rules}")
            resultPd = apply_km_anonymity_r(dataPd, quasi_identifiers, sensitive_column,k_value, m_value, hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'differential_privacy':
            if not epsilon:
                return jsonify({'error': 'Epsilon value is required for differential privacy'}), 400
            resultPd = apply_differential_privacy(dataPd, epsilon, quasi_identifiers,hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
