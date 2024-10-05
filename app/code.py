from pyarx import ARX

data = [
    ['ID', 'Age', 'Gender', 'Disease'],
    ['1', '34', 'M', 'Flu'],
    ['2', '45', 'F', 'Cold'],
    # 更多数据...
]

arx = ARX(data)
# 应用k-匿名性
arx.apply_k_anonymity(3)
anonymized_data = arx.get_result()
print(anonymized_data)
