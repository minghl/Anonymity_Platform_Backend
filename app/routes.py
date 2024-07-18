from flask import Blueprint, request, jsonify
import jpype
import pandas as pd
import os

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
    print("Successfully loaded ARX classes.")
except Exception as e:
    print("Failed to load ARX classes:", e)

def anonymize_data(file_path):
    # 读取上传的CSV文件
    df = pd.read_csv(file_path)

    # 创建ARX数据对象
    data = Data.create()
    columns = list(df.columns)
    data.add(jpype.JArray(jpype.JString)(columns))  # 添加列名
    for index, row in df.iterrows():
        data.add(jpype.JArray(jpype.JString)(list(row.astype(str))))  # 将每行转换为字符串数组

    # 设置准标识符
    data.getDefinition().setAttributeType("age", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("gender", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE)
    data.getDefinition().setAttributeType("zipcode", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE)

    # 创建ARX配置
    config = ARXConfiguration.create()
    config.addPrivacyModel(KAnonymity(1))  # 降低k-匿名性到1

    # 创建并执行ARX匿名化器
    anonymizer = ARXAnonymizer()
    result = anonymizer.anonymize(data, config)

    # 检查匿名化结果的状态
    if result.getGlobalOptimum() is None:
        raise ValueError("No global optimum found for the given k-anonymity model.")

    # 获取匿名化后的数据
    handle = result.getOutput()

    if handle is None:
        raise ValueError("Anonymization failed to produce output.")

    # 转换为DataFrame
    anon_data = []
    iterator = handle.iterator()
    while iterator.hasNext():
        anon_data.append(iterator.next())

    anon_df = pd.DataFrame(anon_data, columns=columns)
    return anon_df

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(bp.root_path, 'uploads', file.filename)
        file.save(file_path)
        
        # 匿名化数据
        try:
            anon_df = anonymize_data(file_path)
            return anon_df.to_json(orient='records')
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

@bp.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from Flask!")
