import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def load_data():
    # 读取数据
    vec = np.load('val_data/val_data.npy')
    y = np.load('val_data/val_label.npy')
    num = len(Counter(y))
    print("测试数据加载成功！")
    print("测试数据类别数量为：", num)
    return vec, y

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(model_path, val_data, val_label):
    try:
        # 尝试加载模型
        model = keras.models.load_model(model_path)
        print("模型：", model_path, "加载成功！")
    except Exception as e:
        print("模型加载失败：", e)
        # 在这里可以添加适当的处理逻辑，例如退出程序或采取其他措施
        exit(1)

    # 打印模型摘要信息
    model.summary()

    # 进行模型评估
    print("*****完成预处理，进行模型评估*****")
    y_pred = model.predict(val_data)
    y_pred = [np.argmax(x) for x in y_pred]

    # 输出评估结果
    print('------------------测试集上得分：------------------------')
    print('*' * 5)
    print('测试集准确率得分:', accuracy_score(val_label, y_pred))
    print('*' * 5)
    print('准确率、召回率、f1-值测试报告如下:\n', classification_report(val_label, y_pred))

    # 绘制混淆矩阵
    plot_confusion_matrix(val_label, y_pred, classes=np.unique(val_label))

if __name__ == '__main__':
    # -------------------设置显存按需分配-----------------
    # 设置显存
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    # 对需要进行限制的GPU进行设置
    #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
    # 查看GPU是否可用
    print("检查GPU是否可用：", tf.test.is_gpu_available())

    # 加载测试数据
    val_data, val_label = load_data()

    # 模型地址名称，如重新训练了模型，请在此处修改地址和名称
    model_path = 'models/cnn_model_epoch-30_valAcc-0.9745454788208008.h5'

    # 进行模型评估
    evaluate_model(model_path, val_data, val_label)
