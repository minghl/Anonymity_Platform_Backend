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

def apply_generalization(dataframe, hierarchy_rules):
    generalized_df = dataframe.copy()
        
    # 打印原始数据类型
    print("原始数据类型：")
    print(generalized_df.dtypes)
        
    for qi, rule in hierarchy_rules.items():
        method = rule.get('method', '').lower()

        if method == 'ordering':
            # 确保列是数值类型
            try:
                generalized_df[qi] = pd.to_numeric(generalized_df[qi], errors='coerce')
            except Exception as e:
                print(f"Error converting {qi} to numeric: {e}")
                return None
            
            # 应用前端提供的层次
            layers = rule.get('layers', [])
            for layer in layers:
                try:
                    min_val = float(layer['min'])  # 确保 min 是 float 类型
                    max_val = float(layer['max'])  # 确保 max 是 float 类型
                except ValueError as ve:
                    print(f"Error converting min or max to float in layer: {layer}")
                    return None
                
                label = f"{min_val}-{max_val}"
                
                # 调试输出：查看转换后的 min_val, max_val 和每一行的值
                print(f"Processing layer: min={min_val}, max={max_val}, label={label}")
                print(f"Column {qi} values (first 5 rows):\n{generalized_df[qi].head()}")

                def check_value(x):
                    # 调试输出：检查 x 的类型和值
                    print(f"Value: {x}, Type: {type(x)}")
                    if pd.notna(x) and isinstance(x, (int, float)):
                        return label if min_val <= x <= max_val else x
                    else:
                        return x
                
                generalized_df[qi] = generalized_df[qi].apply(check_value)
            
            # 打印转换后是否有 NaN 值
            if generalized_df[qi].isna().sum() > 0:
                print(f"Warning: {qi} column contains NaN values after conversion.")
        

        elif method == 'dates':
            # 确保列是日期类型，无法转换的将变为 NaN
            try:
                generalized_df[qi] = pd.to_datetime(generalized_df[qi], errors='coerce')
            except Exception as e:
                print(f"Error converting {qi} to datetime: {e}")
                return None
            
            layers = rule.get('layers', [])
            for layer in layers:
                min_date = pd.to_datetime(layer['min'])
                max_date = pd.to_datetime(layer['max'])
                label = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                
                # 调试输出 min_date 和 max_date，确保它们是 Timestamp 类型
                print(f"min_date: {min_date}, max_date: {max_date}, label: {label}")

                # 检查每个值的类型，并确保它们是 Timestamp 类型
                def check_date_value(x):
                    print(f"Value: {x}, Type: {type(x)}")  # 输出 x 的类型
                    if pd.notna(x) and isinstance(x, pd.Timestamp):
                        return label if min_date <= x <= max_date else x
                    else:
                        return x
                
                generalized_df[qi] = generalized_df[qi].apply(check_date_value)
            
            # 打印转换后是否有 NaN 值
            if generalized_df[qi].isna().sum() > 0:
                print(f"Warning: {qi} column contains NaN values after conversion.")

        elif method == 'masking':
            # 获取掩码字符串，例如 "aa***"
            masking_string = rule.get('maskingString', 'Masked')
            
            # 计算掩码字符串中 * 的位置和数量
            num_stars = masking_string.count('*')
            non_masked_part = masking_string.replace('*', '')
            
            # 应用掩码规则
            def apply_masking(value):
                if value is None or pd.isna(value):
                    return 'Masked'
                value_str = str(value)
                # 如果原始值长度不足以掩码，返回全 *
                if len(value_str) <= len(non_masked_part):
                    return '*' * len(value_str)
                # 保留前面的非掩码部分，后面用 * 覆盖
                return value_str[:len(non_masked_part)] + '*' * num_stars
            
            generalized_df[qi] = generalized_df[qi].apply(apply_masking)
                
        elif method == 'category':
            # 获取该列中的唯一值
            unique_values = dataframe[qi].unique()
            
            # 为每个唯一值生成类型标签
            hierarchy = {val: f"type{i+1}" for i, val in enumerate(unique_values) if pd.notna(val)}
            hierarchy['Masked'] = "Unknown"
            
            # 应用 hierarchy 到数据框中的对应列
            generalized_df[qi] = generalized_df[qi].map(hierarchy)

    print("泛化后的数据框:")
    print(generalized_df.head())
    return generalized_df


def check_m_diversity(group, sensitive_column, m_value):
    sensitive_values = group[sensitive_column].values
    value_counts = Counter(sensitive_values)
    return len(value_counts) >= m_value

def apply_km_anonymity_r(dataframe, quasi_identifiers, sensitive_column, k_value, m_value, hierarchy_rules):
    try:
        # 第一步：生成通用化后的数据
        generalized_df = apply_generalization(dataframe, hierarchy_rules)  # 应用前端传来的分层规则

        print(f"Generalized DataFrame before R processing (k={k_value}):")
        print(generalized_df.head())

        # 转换为 R 的数据格式
        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 使用 sdcMicro 进行 k-匿名处理
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        print(f"Anonymized DataFrame from R (k={k_value}):")
        print(anonymized_df.head())

        # 第二步：检查 m-多样性
        # 根据 quasi_identifiers 进行分组，并检查每个组的敏感属性是否满足 m-多样性
        km_anonymized_groups = []
        for _, group in anonymized_df.groupby(quasi_identifiers):
            # 检查每个组是否满足 k-匿名性和 m-多样性
            if len(group) >= k_value and check_m_diversity(group, sensitive_column, m_value):
                km_anonymized_groups.append(group)
            else:
                # 如果不满足 m-多样性，可以进一步泛化或抑制处理
                print(f"Group does not meet m-diversity (m={m_value}), additional suppression required.")

        # 将满足 km-匿名性和 m-多样性的组重新合并为最终结果
        if km_anonymized_groups:
            km_anonymized_df = pd.concat(km_anonymized_groups)
        else:
            km_anonymized_df = pd.DataFrame()  # 如果没有满足条件的组

        print(f"KM-Anonymized DataFrame (k={k_value}, m={m_value}):")
        print(km_anonymized_df.head())

        return km_anonymized_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def apply_k_anonymity_r(dataframe, quasi_identifiers, k_value, hierarchy_rules):
    try:
        # 第一步：使用前端提供的分层规则进行泛化
        generalized_df = apply_generalization(dataframe, hierarchy_rules)

        print(f"R 处理前的泛化数据框 (k={k_value}):")
        print(generalized_df.head())

        # 记录原始数据中的 NaN 位置
        original_nan_mask = generalized_df[quasi_identifiers].isna()

        # 转换为 R 数据框格式
        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 使用 sdcMicro 进行 k-匿名处理
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        # 第二步：移除 quasi-identifiers 中含有 'Masked' 或空值的行
        mask = anonymized_df[quasi_identifiers].apply(lambda x: x.str.contains("Masked", na=False)) | anonymized_df[quasi_identifiers].isna()

        # 删除任何 quasi-identifier 为 'Masked' 或 NaN 的行
        anonymized_df = anonymized_df[~mask.any(axis=1)]

        # 第三步：删除不满足 k 匿名性的记录
        equivalence_class_size = anonymized_df.groupby(quasi_identifiers).size()

        # 找到小于 k 的等价类
        small_classes = equivalence_class_size[equivalence_class_size < k_value].index
        print(small_classes.head(),'ssss')
        # 删除这些小等价类的行
        anonymized_df = anonymized_df[~anonymized_df[quasi_identifiers].apply(tuple, axis=1).isin(small_classes)]

        print(f"经过 R 处理后 (k={k_value}) 的匿名数据框，移除了小于 k 的等价类及 'Masked' 行:")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"发生错误: {e}")
        return None
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# 修改后的计算 KL 散度部分
def apply_t_closeness_r(dataframe, quasi_identifiers, sensitive_attribute, k_value, t_value, hierarchy_rules):
    try:
        # Step 1: Apply k-anonymity as the base for t-closeness
        generalized_df = apply_generalization(dataframe, hierarchy_rules)  # 应用分层规则
        
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
        overall_distribution = calculate_distribution(dataframe, sensitive_attribute)

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
        # Step 1: 应用分层规则，生成通用化后的数据
        generalized_df = apply_generalization(dataframe, hierarchy_rules)

        print(f"Generalized DataFrame before R processing (k={k_value}, l={l_value}):")
        print(generalized_df.head())

        # 记录原始数据中的 NaN 位置
        original_nan_mask = generalized_df[quasi_identifiers].isna()

        # Step 2: 转换为 R 数据格式，进行 k-匿名处理
        r_dataframe = pandas2ri.py2rpy(generalized_df)
        quasi_identifiers_r = robjects.StrVector(quasi_identifiers)

        # 使用 sdcMicro 进行 k-anonymity
        sdc_obj = sdcmicro.createSdcObj(dat=r_dataframe, keyVars=quasi_identifiers_r)
        anonymized_sdc = sdcmicro.localSuppression(sdc_obj, k=k_value)

        # 提取匿名化后的数据
        anonymized_data = robjects.r['extractManipData'](anonymized_sdc)
        anonymized_df = pandas2ri.rpy2py(anonymized_data)

        # Step 3: 删除准标识符中包含 'Masked' 或 NaN 值的行
        mask = anonymized_df[quasi_identifiers].apply(lambda x: x.str.contains("Masked", na=False)) | anonymized_df[quasi_identifiers].isna()
        anonymized_df = anonymized_df[~mask.any(axis=1)]

        # Step 4: 删除不满足 k-匿名性的记录
        equivalence_class_size = anonymized_df.groupby(quasi_identifiers).size()

        # 找到小于 k 的等价类
        small_classes = equivalence_class_size[equivalence_class_size < k_value].index

        # 删除这些小等价类对应的行
        anonymized_df = anonymized_df[~anonymized_df[quasi_identifiers].apply(tuple, axis=1).isin(small_classes)]

        # Step 5: 检查 l-diversity 约束
        equivalence_classes = anonymized_df.groupby(quasi_identifiers)

        # 检查每个等价类的 l-diversity
        for name, group in equivalence_classes:
            sensitive_values = group[sensitive_attribute].unique()

            # 如果敏感值的数量小于 l，则删除这些行
            if len(sensitive_values) < l_value:
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


# def apply_differential_privacy(dataframe, epsilon, delta, quasi_identifiers, hierarchy_rules, budget, suppression_threshold):
#     try:
#         # 创建一个新的 DataFrame 用于存储差分隐私处理后的数据
#         anonymized_df = dataframe.copy()
        
#         # 使用传入的预算
#         privacy_budget = budget / len(quasi_identifiers)  # 按 quasi_identifiers 平分预算

#         for qi in quasi_identifiers:
#             if qi in dataframe.columns:
#                 # 检查数据类型
#                 rule = hierarchy_rules.get(qi, None)

#                 if rule and rule.get('method') == 'dates':
#                     # 日期类型处理
#                     days_since_epoch = (pd.to_datetime(dataframe[qi], errors='coerce') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                    
#                     # 添加噪声，使用 epsilon 和预算
#                     noise = np.random.laplace(0, 1/(epsilon * privacy_budget), size=days_since_epoch.shape)
#                     noisy_days = days_since_epoch + noise
                    
#                     # 将时间戳转换回日期格式，只保留年月日，并转化为字符串
#                     anonymized_df[qi] = pd.to_datetime(noisy_days, origin='1970-01-01', unit='D').dt.date
#                     anonymized_df[qi] = anonymized_df[qi].astype(str)
                    
#                     # 应用前端传来的层次分组规则
#                     layers = rule.get('layers', [])
#                     for layer in layers:
#                         min_date = pd.to_datetime(layer['min'])
#                         max_date = pd.to_datetime(layer['max'])
#                         label = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                        
#                         # 调试输出 min_date 和 max_date，确保它们是 Timestamp 类型
#                         print(f"min_date: {min_date}, max_date: {max_date}, label: {label}")

#                         # 检查每个值的类型，并确保它们是 Timestamp 类型
#                         def check_date_value(x):
#                             print(f"Value: {x}, Type: {type(x)}")  # 输出 x 的类型
#                             if pd.notna(x):
#                                 try:
#                                     # 尝试将日期字符串转换回时间戳进行比较
#                                     date_value = pd.to_datetime(x)
#                                     return label if min_date <= date_value <= max_date else x
#                                 except Exception as e:
#                                     print(f"Error converting {x} to datetime: {e}")
#                                     return x
#                             else:
#                                 return x
                        
#                         # 应用层次规则到扰动后的日期
#                         anonymized_df[qi] = anonymized_df[qi].apply(check_date_value)
                    
#                     # 打印转换后是否有 NaN 值
#                     if anonymized_df[qi].isna().sum() > 0:
#                         print(f"Warning: {qi} column contains NaN values after conversion.")
       
#                 elif rule and rule.get('method') == 'ordering' and np.issubdtype(dataframe[qi].dtype, np.number):
#                     # 对数值型数据应用差分隐私
#                     column_data = pd.to_numeric(dataframe[qi], errors='coerce')
#                     column_data = column_data.dropna()  # 删除空值
#                     if column_data.empty:
#                         print(f"Column '{qi}' is empty after conversion to numeric. Skipping this column.")
#                         continue
                    
#                     # 添加噪声，使用 epsilon 和预算
#                     noise = np.random.laplace(0, 1/(epsilon * privacy_budget), size=column_data.shape)
#                     anonymized_df[qi] = column_data + noise
                    
#                     layers = rule.get('layers', [])
#                     for layer in layers:
#                         try:
#                             min_val = float(layer['min'])  # 确保 min 是 float 类型
#                             max_val = float(layer['max'])  # 确保 max 是 float 类型
#                         except ValueError as ve:
#                             print(f"Error converting min or max to float in layer: {layer}")
#                             return None
                        
#                         label = f"{min_val}-{max_val}"
                        
#                         # 调试输出：查看转换后的 min_val, max_val 和每一行的值
#                         print(f"Processing layer: min={min_val}, max={max_val}, label={label}")
#                         print(f"Column {qi} values (first 5 rows):\n{anonymized_df[qi].head()}")

#                         def check_value(x):
#                             # 调试输出：检查 x 的类型和值
#                             print(f"Value: {x}, Type: {type(x)}")
#                             if pd.notna(x) and isinstance(x, (int, float)):
#                                 return label if min_val <= x <= max_val else x
#                             else:
#                                 return x
                        
#                         anonymized_df[qi] = anonymized_df[qi].apply(check_value)
                    
#                     # 打印转换后是否有 NaN 值
#                     if anonymized_df[qi].isna().sum() > 0:
#                         print(f"Warning: {qi} column contains NaN values after conversion.")

#                 elif rule and rule.get('method') == 'category':
#                     # 对类别数据进行扰乱
#                     categories = dataframe[qi].unique()
#                     noise_probability = 1 / (1 + np.exp(epsilon))  # 根据 epsilon 决定扰乱概率
                    
#                     def perturb_category(value):
#                         if np.random.rand() < noise_probability:
#                             # 随机选择一个不同类别
#                             available_categories = [cat for cat in categories if cat != value]
#                             return np.random.choice(available_categories)
#                         return value
                    
#                     perturbed_categories = dataframe[qi].apply(perturb_category)
                    
#                     # 将类别映射为 type1, type2 等标签
#                     category_mapping = {val: f"type{i+1}" for i, val in enumerate(perturbed_categories.unique())}
#                     anonymized_df[qi] = perturbed_categories.map(category_mapping)

#                 elif rule and rule.get('method') == 'masking':
#                     # 使用传入的掩码字符串进行掩码处理
#                     masking_string = rule.get('maskingString', '***')
#                     num_stars = masking_string.count('*')
#                     non_masked_part = masking_string.replace('*', '')

#                     def mask_value(value):
#                         value_str = str(value)
#                         if len(value_str) <= len(non_masked_part):
#                             return '*' * len(value_str)
#                         return value_str[:len(non_masked_part)] + '*' * num_stars
                    
#                     anonymized_df[qi] = dataframe[qi].apply(mask_value)

#                 else:
#                     # 非数值类型和日期类型不处理
#                     print(f"Column '{qi}' does not match any known rule. Skipping this column.")
        
#         # 对数据应用抑制，如果超出抑制阈值
#         anonymized_df = anonymized_df.applymap(lambda x: None if np.random.rand() < suppression_threshold else x)

#         # 删除含有 None 的行，表示不符合匿名化要求的数据
#         anonymized_df.dropna(inplace=True)

#         print(f"Differentially Private DataFrame (epsilon={epsilon}, delta={delta}):")
#         print(anonymized_df.head())

#         return anonymized_df

#     except Exception as e:
#         import traceback
#         print("Full traceback of the error:")
#         traceback.print_exc()
#         raise

def apply_differential_privacy(dataframe, epsilon, delta, quasi_identifiers, sensitive_columns, hierarchy_rules, budget, suppression_threshold):
    try:
        # 创建一个新的 DataFrame 用于存储差分隐私处理后的数据
        anonymized_df = dataframe.copy()
        
        # 使用传入的预算
        privacy_budget = budget / len(quasi_identifiers)  # 按 quasi_identifiers 平分预算

        for qi in quasi_identifiers:
            if qi in dataframe.columns:
                # 检查数据类型
                rule = hierarchy_rules.get(qi, None)

                if rule and rule.get('method') == 'dates':
                    # 日期类型处理
                    days_since_epoch = (pd.to_datetime(dataframe[qi], errors='coerce') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                    
                    # 添加噪声，使用 epsilon 和预算
                    noise = np.random.laplace(0, 1/(epsilon * privacy_budget), size=days_since_epoch.shape)
                    noisy_days = days_since_epoch + noise
                    
                    # 将时间戳转换回日期格式，只保留年月日，并转化为字符串
                    anonymized_df[qi] = pd.to_datetime(noisy_days, origin='1970-01-01', unit='D').dt.date
                    anonymized_df[qi] = anonymized_df[qi].astype(str)
                    
                    # 应用前端传来的层次分组规则
                    layers = rule.get('layers', [])
                    for layer in layers:
                        min_date = pd.to_datetime(layer['min'])
                        max_date = pd.to_datetime(layer['max'])
                        label = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                        
                        # 应用层次规则到扰动后的日期
                        anonymized_df[qi] = anonymized_df[qi].apply(lambda x: label if pd.to_datetime(x) >= min_date and pd.to_datetime(x) <= max_date else x)

                    if anonymized_df[qi].isna().sum() > 0:
                        print(f"Warning: {qi} column contains NaN values after conversion.")
       
                elif rule and rule.get('method') == 'ordering' and np.issubdtype(dataframe[qi].dtype, np.number):
                    # 对数值型数据应用差分隐私
                    column_data = pd.to_numeric(dataframe[qi], errors='coerce')
                    column_data = column_data.dropna()  # 删除空值
                    
                    layers = rule.get('layers', [])
                    for layer in layers:
                        try:
                            min_val = float(layer['min'])
                            max_val = float(layer['max'])
                        except ValueError as ve:
                            print(f"Error converting min or max to float in layer: {layer}")
                            return None
                        
                        label = f"{min_val}-{max_val}"
                        
                        # 应用层次规则
                        anonymized_df[qi] = anonymized_df[qi].apply(lambda x: label if min_val <= x <= max_val else x)

                    if anonymized_df[qi].isna().sum() > 0:
                        print(f"Warning: {qi} column contains NaN values after conversion.")

                elif rule and rule.get('method') == 'category':
                    # 对类别数据应用分层规则
                    layers = rule.get('layers', [])
                    category_mapping = {val: f"type{i+1}" for i, val in enumerate(dataframe[qi].unique())}
                    anonymized_df[qi] = dataframe[qi].map(category_mapping)

                elif rule and rule.get('method') == 'masking':
                    masking_string = rule.get('maskingString', '***')
                    num_stars = masking_string.count('*')
                    non_masked_part = masking_string.replace('*', '')

                    def mask_value(value):
                        value_str = str(value)
                        if len(value_str) <= len(non_masked_part):
                            return '*' * len(value_str)
                        return value_str[:len(non_masked_part)] + '*' * num_stars
                    
                    anonymized_df[qi] = dataframe[qi].apply(mask_value)

                else:
                    print(f"Column '{qi}' does not match any known rule. Skipping this column.")
        
        # 处理 sensitive 列（如 REASONDESCRIPTION）
        for sensitive in sensitive_columns:
            if sensitive in dataframe.columns:
                rule = hierarchy_rules.get(sensitive, None)

                if rule and rule.get('method') == 'category':
                    # 对敏感属性直接应用分层规则
                    layers = rule.get('layers', [])
                    category_mapping = {val: f"sensitive_type{i+1}" for i, val in enumerate(dataframe[sensitive].unique())}
                    anonymized_df[sensitive] = dataframe[sensitive].map(category_mapping)

        # 对数据应用抑制
        anonymized_df = anonymized_df.applymap(lambda x: None if np.random.rand() < suppression_threshold else x)
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
            if resultPd is None:
                raise ValueError("Anonymization process returned no result.")
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
            resultPd = apply_differential_privacy(dataPd, epsilon, float(delta), quasi_identifiers, sensitive_column,hierarchy_rules, float(budget), suppression_threshold)
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
