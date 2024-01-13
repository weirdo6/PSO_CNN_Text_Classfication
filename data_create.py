import os
import re
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import collections
import time
from gensim.models import Word2Vec
import numpy as np


def read_book_path(path):
    """
    读取书本内容
    :param path:
    :return:
    """
    book_msg_list = []
    for root, dirs, files in os.walk(path):
        # 获取作者
        for author_dir in dirs:
            # 子文件夹目录
            file_path = os.path.join(root, author_dir)
            for root_2, dirs_2, files_2 in os.walk(file_path):
                for file in files_2:
                    # 获取书本地址
                    book_path = os.path.join(file_path, file)
                    # 获取书本名称
                    book_name = str(file.split('.txt')[0])
                    with open(book_path, "r", encoding='utf-8') as f:  # 打开文件
                        try:
                            txt_data = f.read()  # 读取文件
                        except UnicodeDecodeError:
                            with open(book_path, "r", encoding='gbk') as f:
                                try:
                                    txt_data = f.read()  # 读取文件
                                except UnicodeDecodeError as e:
                                    print(e)
                                    print("错误文件：" + str(book_path))
                                # gbk编码
                                else:
                                    book_msg = [author_dir, book_name, txt_data]
                                    book_msg_list.append(book_msg)
                        # utf-8编码
                        else:
                            book_msg = [author_dir, book_name, txt_data]
                            book_msg_list.append(book_msg)
    return book_msg_list


def words_regularized(text):
    """
    过滤特殊符号以及还原常见缩写单词
    :param text:原始文本
    :return:处理后文本
    """
    # 过滤特殊符号
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    new_text = pat_letter.sub(' ', text).strip().lower()
    # 缩进形式转换
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


def words_rate(data_in, path, feature_num):
    """
    文本预处理+词频统计+特征构建
    :param feature_num:
    :param path: 输出文件夹
    :param data_in:输入dataframe
    :return:
    """
    # 每本书的词频统计
    words_count_list = []
    all_words_list = []
    # 完成每本书词频统计，将所有单词加入：all_words_list
    for i in tqdm(range(len(data_in))):
        content = data_in['msg'][i]
        book_name = data_in['book_name'][i]
        label = data_in['label'][i]
        # 文本预处理
        new_content = words_regularized(content)
        # 词频统计
        words_list = new_content.split()
        words_count = collections.Counter(words_list)
        words_count_list.append([label, book_name, words_count])
        # 将所有单词加入全量词表
        for word in words_list:
            all_words_list.append(word)
    print("完成书籍词汇统计！")
    # 完成所有书籍的词频统计
    all_words_count = collections.Counter(all_words_list)
    # 构建特征列
    df_list = ['label', 'book_name']
    all_words_sorted = sorted(all_words_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in tqdm(range(len(all_words_sorted))):
        df_list.append(str(all_words_sorted[i][0]))
    # 去重
    df_list = list(dict.fromkeys(df_list))
    # 选取前feature_num个单词作为特征
    df_list = df_list[:feature_num + 2]
    # 删除label和book_name
    words_select_list = df_list[2:]
    print("完成特征构建！")
    # 延迟
    time.sleep(1)
    # 填充数据
    df_out_list = []
    for i in tqdm(range(len(words_count_list))):
        # 获取每一行数据
        df_count_list = []
        word_count = words_count_list[i][2]
        book_name = str(words_count_list[i][1])
        label = str(words_count_list[i][0])
        # 加入作者和书籍
        df_count_list.append(label)
        df_count_list.append(book_name)
        # 添加类标数据
        for word in words_select_list:
            # 按行顺序获取value值
            if word in word_count:
                value = word_count[word]
            else:
                value = 0
            # 添加value值
            df_count_list.append(value)
        # 添加一行数据
        df_out_list.append(df_count_list)
    # 构建特征DataFrame
    data_out = pd.DataFrame(df_out_list, columns=df_list)
    print("完成数据填充！")
    # 将所有没有统计数据的列填充0
    data_out.fillna(value=0, inplace=True)
    # 输出预处理文件
    data_out.to_csv(os.path.join(path, 'book_data.csv'), encoding='utf-8-sig', index=False)
    print("完成数据预处理，输出地址为：" + str(os.path.join(path, 'book_data.csv')))
    return data_out


def word_2_vec(data_in, words_num=2000, vec_num=128):
    """
    构建书本-文字特征向量
    :param vec_num: 单词的文本向量维度大小
    :param words_num:段落大小
    :param data_in:输入数据，格式为list：作者，书名，内容
    :return:
    """
    label_list = []
    name_list = []
    word_split_list = []
    # 获取所有文本，文本分割
    for i in tqdm(range(len(data_in))):
        label = data_in[i][0]
        book_name = data_in[i][1]
        content = data_in[i][2]
        # 文本处理
        new_content = words_regularized(content)
        words_list = new_content.split()
        # 文本截取
        words_list = words_list[:len(words_list) - len(words_list) % words_num]
        # 将文本按定义的维度进行截取
        for e in range(1, int(len(words_list) / words_num) + 1):
            word_split_list.append(words_list[(e - 1) * words_num:e * words_num])
            label_list.append(label)
            name_list.append(book_name)
    time.sleep(1)
    print("完成文本预处理，共计获取：", len(word_split_list), "个段落。")
    time_s = datetime.now()
    print("****开始预训练词向量，此处预计耗时20秒（根据文本多少变化）")
    # 词向量训练
    # model = Word2Vec(sentences=word_split_list, vector_size=vec_num, min_count=1)
    time_e = datetime.now()
    time_cql = int((time_e - time_s).total_seconds())
    model = Word2Vec.load('models/word2vec.model')
    # model.save('models/word2vec.model')
    print("完成文本词向量特征预训练,耗时：", time_cql, "秒。", "预训练词向量保存地址：models/word2vec.model")
    # 完成特征构建
    words_vec_list = []
    for i in tqdm(range(len(word_split_list))):
        content = word_split_list[i]
        vec_list = []
        for word in content:
            vec = model.wv[word]
            vec_list.append(vec)
        words_vec_list.append(vec_list)
    time.sleep(1)
    print("完成特征构建。")
    return words_vec_list, label_list, name_list


def label_encode(data_in):
    # 对label进行编码
    data_in = data_in.tolist()
    le_credit_level = LabelEncoder().fit(data_in)
    label_cl = le_credit_level.transform(data_in)
    label_counts = len(collections.Counter(label_cl))
    data_out = np.asarray(label_cl, np.int32)
    print("编码后类别个数:", label_counts)
    print("编码后类标数量:", collections.Counter(data_out))
    lb_list = le_credit_level.classes_
    np.save('train_data/lb_list.npy', lb_list)
    return data_out


if __name__ == '__main__':
    # 书籍文件夹，结构为：‘文件名’/'作者名文件夹'/‘书籍txt文件’。暂只支持txt文件，其他文件不支持
    my_path = 'book_data'
    # 预处理后数据输出文件夹
    out_path = 'train_data'

    data = read_book_path(my_path)
    # """
    # 词频统计形成词频特征
    # """
    # # 定义选取的单词数量作为特征，此处暂定为2w个
    # words_feature = 20000
    # data_pd = pd.DataFrame(data, columns=['label', 'book_name', 'msg'])
    # book_data = words_rate(data_pd, out_path, words_feature)
    """
    构建词特征向量
    """
    d_vec, d_label, d_name = word_2_vec(data)
    np_vec, label, book_name = np.asarray(d_vec, np.float32), np.asarray(d_label), np.asarray(d_name)
    print("训练样本结构：", np_vec.shape)
    np.save('train_data/train_vec.npy', np_vec)
    np.save('train_data/label.npy', label)
    np.save('train_data/name.npy', book_name)

    # 对label进行编码用于训练
    train_label = label_encode(label)
    np.save('train_data/train_label.npy', train_label)
    print("完成特征工程。训练数据保存地址：/train_data/")
