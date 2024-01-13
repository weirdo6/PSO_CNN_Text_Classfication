import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from n_PSO import PSO
import n_model as md

def load_data(data_path='train_data'):
    vec = np.load(os.path.join(data_path, 'train_vec.npy'))
    y = np.load(os.path.join(data_path, 'train_label.npy'))
    num = len(Counter(y))
    print("类别数量为：", num)
    return vec, y, num

def create_train_data(x, y, ratio=0.9):
    num_example = x.shape[0]
    arr = np.arange(num_example)
    np.random.seed(99)
    np.random.shuffle(arr)
    arr_data = x[arr]
    arr_label = y[arr]
    s = int(num_example * ratio)
    x_train, y_train = arr_data[:s], arr_label[:s]
    x_val, y_val = arr_data[s:], arr_label[s:]
    print("训练集shape", x_train.shape)
    print("训练集类别：", Counter(y_train))
    print("测试集shape", x_val.shape)
    print("测试集类别：", Counter(y_val))
    return x_train, y_train, x_val, y_val

def plot_history(history):
    # 绘制损失率
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    data, label, label_count = load_data()

    # 生成训练集测试集
    train_data, train_label, val_data, val_label = create_train_data(data, label)

    # 输入数据的shape值，用于模型构造
    a_shape, b_shape = data.shape[1], data.shape[2]

    # 模型参数
    model_param = {
        "a_shape": a_shape,
        "b_shape": b_shape,
        "label_count": label_count,
        "data": train_data,
        "label": train_label
    }

    """
    用粒子群优化算法对训练模型初始化参数进行优化
    """
    # 设置粒子群优化参数
    dim, size, iter_num = 1, 5, 20
    x_max, x_min, max_vel = 0.01, 0.00001, 0.0005
    pso_param = {
        "dim": dim,
        "size": size,
        "iter_num": iter_num,
        "x_max": x_max,
        "x_min": x_min,
        "max_vel": max_vel
    }

    # 实例化粒子群算法
    pso = PSO(model_param, pso_param)

    # 寻找最优解
    best_err, best_learn_rate = pso.update()
    print("粒子群优化后最优准确率为:", 1 - best_err)
    print("粒子群优化后最优初始化learning_rate:", best_learn_rate)

    # 保存pso优化后参数
    pso_out_param_path = 'models/pso_out_param.json'
    best_param = {
        "acc": 1 - best_err,
        "learn_rate": best_learn_rate,
    }
    with open(pso_out_param_path, 'w') as file:
        json.dump(best_param, file)

    """
    使用最优化初始参数进行训练
    """
    # 模型训练
    cnn_model = md.cnn_model(label_count, data_shape=(a_shape, b_shape))
    cnn_model = cnn_model.model_create(best_learn_rate)

    # 设置 TensorBoard 回调
    log_dir = os.path.join("logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 训练模型，并加入 TensorBoard 回调
    history = cnn_model.fit(train_data, train_label, epochs=30, batch_size=8, validation_split=0.1, callbacks=[tensorboard_callback])

    # 保存模型
    model_path = f'models/cnn_model_epoch-{history.epoch[-1] + 1}_valAcc-{history.history["val_accuracy"][-1]}.h5'
    cnn_model.save(model_path)
    print("完成模型训练，保存地址：", model_path)

    # 保存测试集
    np.save('val_data/val_data.npy', val_data)
    np.save('val_data/val_label.npy', val_label)
    print("测试集保存地址：/val_data/")

    # 绘制损失率和准确率曲线
    plot_history(history)

if __name__ == '__main__':
    main()
