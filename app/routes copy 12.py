from flask import Blueprint, request, jsonify
import pandas as pd
import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from collections import Counter
from scipy.stats import entropy

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
        print(f"h1{hierarchy_rules},h2{hierarchies}")
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

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# 修改后的计算 KL 散度部分
def apply_t_closeness_r(dataframe, quasi_identifiers, sensitive_attribute, k_value, t_value, hierarchy_rules):
    try:
        # Step 1: Apply k-anonymity as the base for t-closeness
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
        mask = anonymized_df[quasi_identifiers].apply(lambda x: x.str.contains("Masked", na=False)) | anonymized_df[quasi_identifiers].isna()
        anonymized_df = anonymized_df[~mask.any(axis=1)]

        # Step 3: Implement t-closeness
        # Calculate the overall distribution of the sensitive attribute
        overall_distribution = dataframe[sensitive_attribute].value_counts(normalize=True)

        # Iterate over each equivalence class
        equivalence_classes = anonymized_df.groupby(quasi_identifiers)
        for name, group in equivalence_classes:
            print(f"Processing equivalence class: {name}")

            # Step 3.1: Ensure there are no missing sensitive attribute values
            if group[sensitive_attribute].isna().sum() > 0:
                print(f"Equivalence class {name} has missing values for sensitive attribute. Skipping this class.")
                continue  # Skip this class if sensitive attribute has missing values

            # Step 3.2: Check if the equivalence class is too small to calculate distribution
            if len(group) < 2:  # You can adjust this threshold as needed
                print(f"Equivalence class {name} is too small for t-closeness. Skipping this class.")
                continue  # Skip this class if it's too small

            # Step 3.3: Calculate the distribution of the sensitive attribute in the equivalence class
            class_distribution = group[sensitive_attribute].value_counts(normalize=True)

            # Step 3.4: Check if the distribution is empty
            if class_distribution.empty:
                print(f"Equivalence class {name} has an empty distribution. Masking sensitive attribute.")
                anonymized_df.loc[group.index, sensitive_attribute] = 'Masked'
                continue  # Skip further processing for this class

            # Step 3.5: Calculate the distance between class and overall distribution
            distance = wasserstein_distance(overall_distribution, class_distribution)
            print(f"t-closeness distance for class {name}: {distance}")

            # Step 3.6: If the distance exceeds the t-value, mask or generalize sensitive attribute
            if distance > t_value:
                print(f"Equivalence class {name} exceeds t-closeness threshold (t={t_value}), masking sensitive attribute.")
                anonymized_df.loc[group.index, sensitive_attribute] = 'Masked'

        # Step 4: Remove rows with 'Masked' values in sensitive attribute after t-closeness processing
        anonymized_df = anonymized_df[anonymized_df[sensitive_attribute] != 'Masked']

        print(f"Anonymized DataFrame after t-closeness processing (k={k_value}, t={t_value}):")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during t-closeness processing: {e}")
        return None

def calculate_distribution(dataframe, sensitive_attribute):
    """
    计算敏感属性的分布。
    """
    return dataframe[sensitive_attribute].value_counts(normalize=True)

def calculate_kl_divergence(p, q):
    """
    计算两个分布之间的KL散度，用于评估t-closeness。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    return entropy(p, q)



def apply_l_diversity_r(dataframe, quasi_identifiers, sensitive_attribute, k_value, l_value, hierarchy_rules):
    try:
        # Step 1: Generate generalized data (same as in k-anonymity)
        hierarchies = create_generalization_hierarchy(dataframe, quasi_identifiers, hierarchy_rules)
        generalized_df = apply_generalization(dataframe, hierarchies)

        print(f"Generalized DataFrame before R processing (k={k_value}, l={l_value}):")
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

        # Step 3: Remove records that don't satisfy k-anonymity
        equivalence_class_size = anonymized_df.groupby(quasi_identifiers).size()

        # Find equivalence classes smaller than k
        small_classes = equivalence_class_size[equivalence_class_size < k_value].index

        # Remove rows corresponding to small equivalence classes
        anonymized_df = anonymized_df[~anonymized_df[quasi_identifiers].apply(tuple, axis=1).isin(small_classes)]

        # Step 4: Implement l-diversity constraint
        equivalence_classes = anonymized_df.groupby(quasi_identifiers)

        # Check l-diversity for each equivalence class
        for name, group in equivalence_classes:
            sensitive_values = group[sensitive_attribute].unique()

            # If the number of distinct sensitive values is less than l, suppress the group
            if len(sensitive_values) < l_value:
                # Optionally, further generalize or remove these rows
                anonymized_df = anonymized_df.drop(group.index)

        print(f"Anonymized DataFrame (k={k_value}, l={l_value}) after applying l-diversity:")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



from diffprivlib.tools import mean
import numpy as np
import pandas as pd
from scipy.stats import entropy
p = [0.1, 0.9]
q = [0.8, 0.2]
print(entropy(p, q))


def apply_differential_privacy(dataframe, epsilon, delta, quasi_identifiers, hierarchy_rules, budget, suppression_threshold):
    try:
        # 创建一个新的 DataFrame 用于存储差分隐私处理后的数据
        anonymized_df = dataframe.copy()
        
        # 使用传入的预算
        privacy_budget = budget / len(quasi_identifiers)  # 按 quasi_identifiers 平分预算

        for qi in quasi_identifiers:
            if qi in dataframe.columns:
                # 检查数据类型
                rule = hierarchy_rules.get(qi, None)
                
                if rule == "dates":
                    # 日期类型处理
                    days_since_epoch = (pd.to_datetime(dataframe[qi], errors='coerce') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                    
                    # 添加噪声，使用 epsilon 和预算
                    noise = np.random.laplace(0, 1/(epsilon * privacy_budget), size=days_since_epoch.shape)
                    noisy_days = days_since_epoch + noise
                    
                    # 将时间戳转换回日期格式，只保留年月日
                    anonymized_df[qi] = pd.to_datetime(noisy_days, origin='1970-01-01', unit='D').dt.date
                    anonymized_df[qi] = anonymized_df[qi].astype(str)
                
                elif rule == "ordering" and np.issubdtype(dataframe[qi].dtype, np.number):
                    # 对数值型数据应用差分隐私
                    column_data = pd.to_numeric(dataframe[qi], errors='coerce')
                    column_data = column_data.dropna()  # 删除空值
                    if column_data.empty:
                        print(f"Column '{qi}' is empty after conversion to numeric. Skipping this column.")
                        continue
                    
                    # 添加噪声，使用 epsilon 和预算
                    noise = np.random.laplace(0, 1/(epsilon * privacy_budget), size=column_data.shape)
                    anonymized_df[qi] = column_data + noise

                elif rule == "category":
                    # 对类别数据进行扰乱
                    categories = dataframe[qi].unique()
                    noise_probability = 1 / (1 + np.exp(epsilon))  # 根据 epsilon 决定扰乱概率
                    
                    def perturb_category(value):
                        if np.random.rand() < noise_probability:
                            # 随机选择一个不同类别
                            available_categories = [cat for cat in categories if cat != value]
                            return np.random.choice(available_categories)
                        return value
                    
                    anonymized_df[qi] = dataframe[qi].apply(perturb_category)

                elif rule == "masking":
                    # 对标记为 "masking" 的列进行掩码处理，保留前两位，后两位加 **
                    def mask_value(value):
                        value_str = str(value)
                        if len(value_str) > 2:
                            return value_str[:-2] + "**"  # 替换后两位为 **
                        return value_str  # 如果长度不足2位，不替换
                    
                    anonymized_df[qi] = dataframe[qi].apply(mask_value)

                else:
                    # 非数值类型和日期类型不处理
                    print(f"Column '{qi}' does not match any known rule. Skipping this column.")
        
        # 对数据应用抑制，如果超出抑制阈值
        anonymized_df = anonymized_df.applymap(lambda x: None if np.random.rand() < suppression_threshold else x)

        # 删除含有 None 的行，表示不符合匿名化要求的数据
        anonymized_df.dropna(inplace=True)

        print(f"Differentially Private DataFrame (epsilon={epsilon}, delta={delta}):")
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
            resultPd = apply_l_diversity_r(dataPd, quasi_identifiers, sensitive_column, k_value,l_value, hierarchy_rules)
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 't-closeness':
            resultPd = apply_t_closeness_r(dataPd, quasi_identifiers, sensitive_column, k_value, t_value, hierarchy_rules)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply t-closeness.'}), 400
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
