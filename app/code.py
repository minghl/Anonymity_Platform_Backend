from pyarxaas import ARXaaS
from pyarxaas.privacy_models import KAnonymity
from pyarxaas.hierarchy import Hierarchy
from pyarxaas.dataset import Dataset
from pyarxaas.attributes import QuasiIdentifier

# 初始化ARXaaS客户端
arxaas = ARXaaS("http://localhost:8080")

# 创建数据集
data = [
    ["age", "gender", "zipcode"],
    ["34", "male", "81667"],
    ["35", "female", "81668"],
    ["36", "female", "81669"]
]

dataset = Dataset(data)

# 创建层次结构
age_hierarchy = Hierarchy([
    ["34", "30-40", "*"],
    ["35", "30-40", "*"],
    ["36", "30-40", "*"]
])

zipcode_hierarchy = Hierarchy([
    ["81667", "8166*", "816**", "81***", "8****", "*****"],
    ["81668", "8166*", "816**", "81***", "8****", "*****"],
    ["81669", "8166*", "816**", "81***", "8****", "*****"]
])

# 定义数据集的准标识符
dataset.set_attribute_type("age", QuasiIdentifier(age_hierarchy))
dataset.set_attribute_type("zipcode", QuasiIdentifier(zipcode_hierarchy))

# 设置k-匿名性模型
k_anonymity = KAnonymity(2)

# 进行数据匿名化
anonymized_dataset = arxaas.anonymize(dataset, k_anonymity)

# 获取匿名化后的数据
print(anonymized_dataset)

