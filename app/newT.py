import jpype
import pandas as pd
import os

# 启动 JVM
def start_jvm():
    if not jpype.isJVMStarted():
        jar_path = os.path.abspath("arx-3.9.1-osx-64.jar")
        jpype.startJVM(classpath=[jar_path])
        print(f"JVM started with JAR path: {jar_path}")

# 停止 JVM
def stop_jvm():
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
        print("JVM stopped.")

def apply_arx(data_path):
    start_jvm()
    try:
        ARXAnonymizer = jpype.JClass("org.deidentifier.arx.ARXAnonymizer")
        ARXData = jpype.JClass("org.deidentifier.arx.Data")  # 检查实际的类名和方法

        # 读取数据
        dataframe = pd.read_csv(data_path)
        print("DataFrame loaded:", dataframe.head())

        # Convert pandas DataFrame to Java Data object
        java_data = ARXData.fromPandas(dataframe)
        print("Converted Data:", java_data)

        # 使用 ARX 进行处理
        anonymizer = ARXAnonymizer()
        # Check method usage
        print("Using ARXAnonymizer instance:", anonymizer)
        # 继续处理...
        
    except Exception as e:
        print(f"Error applying ARX: {e}")
    finally:
        stop_jvm()

apply_arx("/Users/liminghao/Documents/Study/Py_Fl_Master/backend/uploads/test_data.csv")
