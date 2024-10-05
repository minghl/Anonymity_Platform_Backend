from flask import Blueprint, request, jsonify
import pandas as pd
import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

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

import json

@bp.route('/anonymize', methods=['POST'])
def anonymize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    privacy_model = request.form.get('privacy_model', 'k-anonymity')
    k_value = int(request.form.get('k', 2))
    
    l_value = request.form.get('l', None)
    if l_value:
        l_value = int(l_value)
    
    t_value = request.form.get('t', None)
    if t_value:
        t_value = float(t_value)

    quasi_identifiers = request.form.get('quasi_identifiers', 'Gender,Age,Zipcode').split(',')
    sensitive_column = request.form.get('sensitive_column', 'Disease')
    
    # 解析 hierarchy_rules JSON 字符串
    hierarchy_rules = json.loads(request.form.get('hierarchy_rules', '{}'))

    file_path = os.path.join(bp.root_path, 'uploads', file.filename)
    file.save(file_path)

    dataPd = pd.read_csv(file_path)

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
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
