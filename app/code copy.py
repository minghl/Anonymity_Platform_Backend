import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# 激活pandas与R的数据帧转换
pandas2ri.activate()

# 导入sdcmicro包
sdcmicro = importr('sdcMicro')

# 创建一个Pandas数据框架作为示例数据
import pandas as pd

data = {
    'Age': [25, 35, 25, 25, 25, 25, 45, 45, 45],
    'Gender': ['Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female'],
    'Zipcode': ['23456', '12345', '12345', '23456', '23456', '23456', '34567', '34567', '34567'],
    'Disease': ['DiseaseA', 'DiseaseB', 'DiseaseC', 'DiseaseA', 'DiseaseB', 'DiseaseA', 'DiseaseA', 'DiseaseA', 'DiseaseB']
}

df = pd.DataFrame(data)

# 将Pandas数据框架转换为R数据框架，并将'Disease'列转换为因子
r_df = pandas2ri.py2rpy(df)
r_df = robjects.r('as.data.frame')(r_df)
r_df[3] = robjects.r('as.factor')(r_df[3])  # 将'Disease'列转换为因子

# 定义准标识符
quasi_identifiers = robjects.StrVector(['Age', 'Gender', 'Zipcode'])

# 使用sdcmicro的createSdcObj函数创建一个SdcMicro对象
sdc_obj = sdcmicro.createSdcObj(
    dat=r_df,
    keyVars=quasi_identifiers
)

# 获取'Disease'列的索引
disease_index = robjects.IntVector([4])  # 假设'Disease'是第4列

# 使用ldiversity进行l-diversity处理
ldiversity_result = sdcmicro.ldiversity(sdc_obj, ldiv_index=disease_index, l_recurs_c=2)

# 查看RS4对象的slotNames
slot_names = robjects.r.slotNames(ldiversity_result)
print(f"Available slots in ldiversity_result: {slot_names}")

# 提取匿名化处理后的数据
anonymized_data = robjects.r.slot(ldiversity_result, 'manipKeyVars')

# 将R对象转换回Pandas数据框
anonymized_df = pandas2ri.rpy2py(anonymized_data)

# 输出结果
print("Anonymized Data:")
print(anonymized_df,anonymized_data)
