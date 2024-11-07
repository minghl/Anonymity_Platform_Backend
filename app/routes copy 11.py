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
        if qi in hierarchy_rules:
            rule = hierarchy_rules[qi].lower()
            if rule == 'ordering':
                min_val, max_val = dataframe[qi].min(), dataframe[qi].max()
                unique_values = dataframe[qi].dropna().unique()
                bins = np.linspace(min_val, max_val, num=min(len(unique_values), 4))
                labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
                hierarchy = {}
                for val in unique_values:
                    bin_index = np.digitize(val, bins) - 1
                    bin_index = min(bin_index, len(labels) - 1)
                    hierarchy[val] = labels[bin_index]

                # 将空值回填为对应的分层区间
                hierarchy['Masked'] = labels[0]  # 默认将空值分配到第一个区间

            elif rule == 'masking':
                # 先填充 NaN 并转换为字符串
                dataframe[qi] = dataframe[qi].fillna('Masked').astype(str)
                
                # 获取填充后的唯一值
                unique_values = dataframe[qi].unique()
                
                # 应用 masking 规则
                hierarchy = {
                    val: (val[:-2] + '**' if len(val) > 2 and val != 'Masked' else 'Masked') 
                    for val in unique_values
                }

                # 确保 'Masked' 被正确映射
                hierarchy['Masked'] = 'Masked'

                print(f"Hierarchy for {qi}: {hierarchy}")
                print(f"DataFrame after masking for {qi}:")
                print(dataframe[qi].head())



            elif rule == 'dates':
                # 首先将日期转换为时间戳
                unique_values = dataframe[qi].unique()
                date_range = pd.to_datetime(unique_values, errors='coerce')

                # 检查是否存在有效日期
                if date_range.notna().sum() > 0:
                    min_timestamp = date_range.min()  # 最早的时间戳
                    max_timestamp = date_range.max()  # 最晚的时间戳

                    # 将日期时间范围划分为 5 个区间
                    bins = pd.date_range(start=min_timestamp, end=max_timestamp, periods=6)  # 6个点划分5个区间

                    hierarchy = {}
                    for val in unique_values:
                        if pd.isna(val):
                            hierarchy[val] = "Unknown"
                        else:
                            timestamp_val = pd.to_datetime(val, errors='coerce')
                            # 根据时间戳将值映射到对应的区间
                            for i in range(len(bins) - 1):
                                if bins[i] <= timestamp_val < bins[i + 1]:
                                    hierarchy[val] = f"{bins[i]} - {bins[i + 1]}"
                                    break
                else:
                    # 如果没有有效的日期，设置为 "Unknown"
                    hierarchy = {val: "Unknown" for val in unique_values}

            elif rule == 'category':
                unique_values = dataframe[qi].unique()
                hierarchy = {val: f"type{i+1}" for i, val in enumerate(unique_values) if pd.notna(val)}
                hierarchy['Masked'] = "Unknown"
        else:
            if np.issubdtype(dataframe[qi].dtype, np.number):
                min_val, max_val = dataframe[qi].min(), dataframe[qi].max()
                unique_values = dataframe[qi].dropna().unique()
                bins = np.linspace(min_val, max_val, num=min(len(unique_values), 4))
                labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
                hierarchy = {}
                for val in unique_values:
                    bin_index = np.digitize(val, bins) - 1
                    bin_index = min(bin_index, len(labels) - 1)
                    hierarchy[val] = labels[bin_index]

                # 回填空值为最接近的分层区间
                hierarchy['Masked'] = labels[0]
            else:
                unique_values = dataframe[qi].unique()
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
        # Step 1: Generate generalized data
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        print(f"Generalized DataFrame before R processing (k={k_value}):")
        print(generalized_df.head())

        # 记录原始数据中的 NaN 位置
        original_nan_mask = generalized_df[quasi_identifiers].isna()

        # Convert to R dataframe format
        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # Use sdcMicro for k-anonymity
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # Extract anonymized data
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        # Step 2: Remove rows with 'Masked' or empty values in quasi-identifiers
        # Check for 'Masked' or NaN values in the quasi-identifiers
        mask = anonymized_df[quasi_identifiers].apply(lambda x: x.str.contains("Masked", na=False)) | anonymized_df[quasi_identifiers].isna()

        # Remove rows where any quasi-identifier is 'Masked' or NaN
        anonymized_df = anonymized_df[~mask.any(axis=1)]

        # Step 3: 删除不满足 k 匿名性的记录
        equivalence_class_size = anonymized_df.groupby(quasi_identifiers).size()

        # 找到小于 k 的等价类
        small_classes = equivalence_class_size[equivalence_class_size < k_value].index

        # 删除这些小等价类的行
        anonymized_df = anonymized_df[~anonymized_df[quasi_identifiers].apply(tuple, axis=1).isin(small_classes)]

        print(f"Anonymized DataFrame from R (k={k_value}) after removing classes with size < k and 'Masked' rows:")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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
        # Step 1: Create generalization hierarchies for the quasi-identifiers
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        
        # Step 2: Apply the generalization to the dataframe
        generalized_df = apply_generalization(dataframe, hierarchies)
        
        # Step 3: Convert pandas DataFrame to R DataFrame
        r_dataframe = pandas2ri.py2rpy(generalized_df)

        # Convert quasi-identifiers to R vector
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # Step 4: Create the SDC object in R
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)

        # Get the index of the sensitive column in R (1-based index)
        sensitive_column_index = dataframe.columns.get_loc(sensitive_column) + 1

        # Step 5: Apply l-diversity with the specified l_value
        ldiversity_result = sdcmicro.ldiversity(sdc_obj, ldiv_index=robjects.IntVector([sensitive_column_index]), l_recurs_c=l_value)

        # Step 6: Extract the anonymized data
        anonymized_data = robjects.r['extractManipData'](ldiversity_result)

        # Step 7: Convert the R DataFrame back to pandas
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

def apply_differential_privacy(dataframe, epsilon, delta, quasi_identifiers, hierarchy_rules, budget, suppression_threshold):
    try:
        # 初始化用于管理预算的变量
        remaining_budget = budget / 100  # 百分比形式，转换为小数
        epsilon_used_per_column = epsilon * remaining_budget / len(quasi_identifiers)

        # 创建一个新的 DataFrame 用于存储差分隐私处理后的数据
        anonymized_df = dataframe.copy()

        # Step 1: 对所有准标识符进行处理
        for qi in quasi_identifiers:
            if qi in dataframe.columns:
                # 检查数据类型
                rule = hierarchy_rules.get(qi, None)

                # Step 1.1: 泛化数值列
                if rule == 'ordering' and np.issubdtype(dataframe[qi].dtype, np.number):
                    min_val = dataframe[qi].min()
                    max_val = dataframe[qi].max()
                    bin_edges = np.arange(np.floor(min_val / 10) * 10, np.ceil(max_val / 10) * 10 + 10, 10)
                    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
                    anonymized_df[qi] = pd.cut(dataframe[qi], bins=bin_edges, labels=labels, include_lowest=True)

                # Step 1.2: 泛化日期列
                elif rule == "dates":
                    dates_converted = pd.to_datetime(dataframe[qi], errors='coerce')
                    anonymized_df[qi] = "Q" + dates_converted.dt.quarter.astype(str)
                    anonymized_df[qi].fillna("Unknown", inplace=True)

                # Step 1.3: 掩码处理
                elif rule == "masking":
                    anonymized_df[qi] = "Masked"

                # Step 2: 差分隐私处理
                if rule in ["ordering", "dates"]:
                    column_data = pd.to_numeric(dataframe[qi], errors='coerce').dropna()
                    if column_data.empty:
                        continue

                    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_used_per_column
                    noise = np.random.normal(0, sigma, size=column_data.shape)
                    noisy_column = column_data + noise

                    suppression_mask = np.abs(noise) < suppression_threshold
                    noisy_column[suppression_mask] = np.nan

                    # 重新泛化 noisy_column
                    if rule == 'ordering':
                        min_val = dataframe[qi].min()
                        max_val = dataframe[qi].max()
                        bin_edges = np.arange(np.floor(min_val / 10) * 10, np.ceil(max_val / 10) * 10 + 10, 10)
                        labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
                        anonymized_df[qi] = pd.cut(noisy_column, bins=bin_edges, labels=labels, include_lowest=True)

                    elif rule == 'dates':
                        dates_converted = pd.to_datetime(noisy_column, errors='coerce')
                        anonymized_df[qi] = "Q" + dates_converted.dt.quarter.astype(str)

                    # Step 3: 将 suppress 的值设置为 `*`
                    if anonymized_df[qi].dtype.name == 'category':
                        if "*" not in anonymized_df[qi].cat.categories:
                            anonymized_df[qi] = anonymized_df[qi].cat.add_categories(['*'])

                    # 将抑制的条目标记为 '*'
                    anonymized_df.loc[suppression_mask, qi] = "*"
                    anonymized_df[qi].fillna("*", inplace=False)

        # Step 4: 确保所有被 suppress 的行都同步在其他准标识符中显示为 `*`
        suppression_rows_mask = anonymized_df[quasi_identifiers].isna().any(axis=1) | anonymized_df[quasi_identifiers].eq("*").any(axis=1)

        # 将所有 NaN 和 `*` 的行对应的所有准标识符列设置为 `*`
        anonymized_df.loc[suppression_rows_mask, quasi_identifiers] = "*"

        print(f"Differentially Private DataFrame with (epsilon={epsilon}, delta={delta}, budget={budget}%):")
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
            delta = request.form.get('delta', '1e-6')  # 默认delta为1e-6
            budget = request.form.get('budget', '100')  # 默认预算为100%
            suppression_threshold = float(request.form.get('suppression_threshold', '0.1'))  # 抑制阈值
            resultPd = apply_differential_privacy(dataPd, epsilon, float(delta), quasi_identifiers, hierarchy_rules, float(budget), suppression_threshold)
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
