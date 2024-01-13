from data_create import read_book_path, word_2_vec
import numpy as np
from tensorflow import keras
from collections import Counter
import tensorflow as tf


def book_pre():
    #gpu_memory = 5120
    ###tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
    # 加载模型
    model_path = 'models/cnn_model_epoch-30_valAcc-0.94545454.h5'
    model = keras.models.load_model(model_path)
    print("模型：", model_path, "。加载成功！")
    my_path = 'book_test'
    data = read_book_path(my_path)
    for i in range(len(data)):
        print("*" * 10)
        print("第：", i + 1, " 本书")
        print("读取的书名：", data[i][1])
        print("读取的作者名：", data[i][0])
    # 选第x本书
    print("*" * 10)
    choose_num = int(input("请输入您想选取进行预测的书的序号。")) - 1
    in_data = [data[choose_num]]
    book_name = data[choose_num][1]
    real_author = data[choose_num][0]
    d_vec, d_label, d_name = word_2_vec(in_data)
    np_vec, label, b_name = np.asarray(d_vec, np.float32), np.asarray(d_label), np.asarray(d_name)
    y_pred = model.predict(np_vec)
    y_pred = [np.argmax(x) for x in y_pred]
    print("完成预测！")
    author_code = max(Counter(y_pred))
    lb_list = np.load('train_data/lb_list.npy')
    author_name = lb_list[author_code]
    print("选取的书籍为：", book_name)
    print("模型预测结果的作者为：", author_name)
    print("实际的作者为：", real_author)


if __name__ == '__main__':
    book_pre()
