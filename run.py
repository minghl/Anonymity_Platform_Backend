from app import create_app
import jpype
import atexit

app = create_app()

# 在应用退出时关闭JVM
def shutdown_jvm():
    if jpype.isJVMStarted():
        jpype.shutdownJVM()

atexit.register(shutdown_jvm)

if __name__ == '__main__':
    app.run(debug=True)