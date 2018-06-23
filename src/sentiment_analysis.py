#!/usr/bin/env python3
# coding: utf-8
# File: sentiment_analysis.py
# Author: lxw
# Date: 17-12-20

"""
基于预训练词向量模型的情感分类
"""

import numpy as np
import time

from collections import Counter
from keras.preprocessing import sequence
from pyfasttext import FastText
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


class Preprocessing:
    def __init__(self):
        start_time = time.time()
        # self.model = FastText("../data/input/models/sg_pyfasttext.bin")  # DEBUG
        self.model = FastText("../data/input/models/880w_fasttext_skip_gram.bin")
        end_time = time.time()
        print(f"Loading word vector model cost: {end_time - start_time:.2f}s")

        # self.vocab_size, self.vector_size = self.model.numpy_normalized_vectors.shape  # OK
        self.vocab_size = self.model.nwords
        self.vector_size = self.model.args.get("dim")
        # self.vector_size:200, self.vocab_size: 925242
        print(f"self.vector_size:{self.vector_size}, self.vocab_size: {self.vocab_size}")

        # 句子的表示形式: {"avg": 向量和的平均, "fasttext": get_numpy_sentence_vector, "concatenate": 向量拼接和补齐}
        self.sentence_vec_type = "fasttext"

        self.MAX_SENT_LEN = 30  # DEBUG: 超参数. self.get_sent_max_length()
        # 100: 50.22%, 80: 50.23%, 70: 50.33%, 60: 55.92%, 50: 69.11%, 40: 68.91%, 36: 69.34%, 30: 69.22%, 20: 69.17%, 10: 67.07%

    def set_sent_vec_type(self, sentence_vec_type):
        assert sentence_vec_type in ["avg", "concatenate", "fasttext"], \
            "sentence_vec_type must be in ['avg', 'fasttext', 'concatenate']"
        self.sentence_vec_type = sentence_vec_type

    def get_sent_max_length(self):  # NOT_USED
        sent_len_counter = Counter()
        max_length = 0
        with open("../data/input/training_set.txt") as f:
            for line in f:
                content = line.strip().split("\t")[1]
                content_list = content.split()
                length = len(content_list)
                sent_len_counter[length] += 1
                if max_length <= length:
                    max_length = length
        sent_len_counter = sorted(list(sent_len_counter.items()), key=lambda x: x[0])
        print(sent_len_counter)
        # [(31, 1145), (32, 1105), (33, 1017), (34, 938), (35, 839), (36, 830), (37, 775), (38, 737), (39, 720), (40, 643), (41, 575), (42, 584), (43, 517), (44, 547), (45, 514), (46, 514), (47, 480), (48, 460), (49, 470), (50, 444), (51, 484), (52, 432), (53, 462), (54, 495), (55, 487), (56, 500), (57, 496), (58, 489), (59, 419), (60, 387), (61, 348), (62, 265), (63, 222), (64, 153), (65, 127), (66, 103), (67, 67), (68, 34), (69, 21), (70, 22), (71, 8), (72, 6), (73, 4), (74, 10), (75, 2), (76, 4), (77, 2), (78, 1), (79, 2), (80, 4), (81, 2), (82, 3), (83, 1), (84, 5), (86, 4), (87, 3), (88, 3), (89, 2), (90, 2), (91, 3), (92, 5), (93, 2), (94, 4), (96, 1), (97, 5), (98, 1), (99, 2), (100, 2), (101, 2), (102, 1), (103, 2), (104, 2), (105, 2), (106, 5), (107, 3), (108, 2), (109, 3), (110, 4), (111, 1), (112, 2), (113, 3), (114, 1), (116, 1), (119, 3), (679, 1)]
        return max_length

    def gen_sentence_vec(self, sentence):
        """
        :param sentence: 
        :return: 
        """
        sentence = sentence.strip()
        if self.sentence_vec_type == "fasttext":
            return self.model.get_numpy_sentence_vector(sentence)

        word_list = sentence.split(" ")
        if self.sentence_vec_type == "concatenate":
            """
            word_len = len(word_list)
            sentence_matrix = np.empty(word_len, dtype=list)
            for idx, word in enumerate(word_list):
                sentence_matrix[idx] = self.model.get_numpy_vector(word)
            """
            sentence_vector = self.model.get_numpy_vector(word_list[0])
            for word in word_list[1:]:
                sentence_vector = np.hstack((sentence_vector, self.model.get_numpy_vector(word)))
            return sentence_vector  # NOTE: 对于concatenate情况, 每个句子的sentence_vector是不一样长的
        else:  # self.sentence_vec_type == "avg":
            sentence_vector = np.zeros(self.vector_size)  # <ndarray>
            # print(f"type(sentence_vector): {type(sentence_vector)}")
            for idx, word in enumerate(word_list):
                # print(f"type(self.model.get_numpy_vector(word)): {type(self.model.get_numpy_vector(word))}")  # <ndarray>
                sentence_vector += self.model.get_numpy_vector(word)
            return sentence_vector / len(word_list)

    def gen_train_val_data(self):
        # 构造训练数据 & 验证数据
        X_train = list()  # <list of ndarray>
        y_train = list()
        for line in open("../data/input/training_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1])
            X_train.append(sent_vector)
            y_train.append(int(line[0]))
        if self.sentence_vec_type == "concatenate":
            # NOTE: 注意，这里的dtype是必须的，否则dtype默认值是"int32", 词向量所有的数值会被全部转换为0
            X_train = sequence.pad_sequences(X_train, maxlen=self.MAX_SENT_LEN * self.vector_size, value=0,
                                             dtype=np.float)

        X_val = list()
        y_val = list()
        for line in open("../data/input/validation_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1])
            X_val.append(sent_vector)
            y_val.append(int(line[0]))
        if self.sentence_vec_type == "concatenate":
            X_val = sequence.pad_sequences(X_val, maxlen=self.MAX_SENT_LEN * self.vector_size, value=0,
                                           dtype=np.float)

        return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)


class SentimentAnalysis:
    def __init__(self):
        self.model_path_prefix="../data/output/models/"
        self.algorithm_name = "nb"
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}.model"

    def pick_algorithm(self, algorithm_name):
        assert algorithm_name in ["nb", "dt", "knn", "svm", "mlp", "cnn", "lstm"], \
            "algorithm_name must be in ['nb', 'dt', 'knn', 'svm', 'mlp', 'cnn', 'lstm']"
        self.algorithm_name = algorithm_name
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}.model"

    def get_model_class(self):
        model_cls = None
        if self.algorithm_name == "nb":  # Naive Bayes
            from sklearn.naive_bayes import GaussianNB
            model_cls = GaussianNB()
        elif self.algorithm_name == "dt":  # Decision Tree
            from sklearn import tree
            model_cls = tree.DecisionTreeClassifier()
        elif self.algorithm_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            model_cls = KNeighborsClassifier()
            tuned_parameters = [{"n_neighbors": range(1, 20)}]
            model_cls = GridSearchCV(model_cls, tuned_parameters, cv=5, scoring="precision_weighted")
        elif self.algorithm_name == "svm":
            from sklearn.svm import SVC
            model_cls = SVC(kernel="linear")
            tuned_parameters = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
                                {"kernel": ["linear"], "C": [1, 10, 100, 1000]}]
            model_cls = GridSearchCV(model_cls, tuned_parameters, cv=5, scoring="precision_weighted")

        return model_cls

    def model_train(self, model_cls, X_train, y_train):
        """
        分类器模型的训练
        :param model_cls: 所使用的算法的类的定义，尚未训练的模型
        :param X_train: 
        :param y_train: 
        :return: 训练好的模型
        """
        model_cls.fit(X_train, y_train)
        return model_cls  # model

    def model_save(self, model):
        """
        分类器模型的保存
        :param model: 训练好的模型对象
        :return: None
        """
        joblib.dump(model, self.model_path)

    def model_evaluate(self, model, X_val, y_val):
        """
        分类器模型的评估
        :param model: 训练好的模型对象
        :param X_val: 
        :param y_val: 
        :return: None
        """
        # model = joblib.load(self.model_path)
        y_val = list(y_val)
        correct = 0
        y_predict = model.predict(X_val)
        print(f"len(y_predict): {len(y_predict)}, len(y_val): {len(y_val)}")
        assert len(y_predict) == len(y_val), "Unexpected Error: len(y_predict) != len(y_val), but it should be"
        for idx in range(len(y_predict)):
            if int(y_predict[idx]) == int(y_val[idx]):
                correct += 1
        score = correct / len(y_predict)
        print(f"'{self.algorithm_name}' Classification Accuray:{score*100:.2f}%")

    def model_predict(self, model, preprocess_obj):
        """
        模型测试
        :param model: 训练好的模型对象
        :param preprocess_obj: Preprocessing类对象
        :return: None
        """
        sentence = "这件 衣服 真的 太 好看 了 ！ 好想 买 啊 "
        sent_vec = np.array(preprocess_obj.gen_sentence_vec(sentence)).reshape(1, -1)  # shape: (1, 1000)
        # print(f"sent_vec: {sent_vec.tolist()}")
        if preprocess_obj.sentence_vec_type == "concatenate":
            # NOTE: 注意，这里的dtype是必须的，否则dtype默认值是'int32', 词向量所有的数值会被全部转换为0
            sent_vec = sequence.pad_sequences(sent_vec, maxlen=preprocess_obj.MAX_SENT_LEN * preprocess_obj.vector_size,
                                              value=0, dtype=np.float)
            # print(f"sent_vec: {sent_vec.tolist()}")
        print(f"'{sentence}': {model.predict(sent_vec)}")  # 0: 正向

        sentence = "这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了"
        sent_vec = np.array(preprocess_obj.gen_sentence_vec(sentence)).reshape(1, -1)
        # print(f"sent_vec: {sent_vec.tolist()}")
        if preprocess_obj.sentence_vec_type == "concatenate":
            sent_vec = sequence.pad_sequences(sent_vec, maxlen=preprocess_obj.MAX_SENT_LEN * preprocess_obj.vector_size,
                                              value=0, dtype=np.float)
            # print(f"sent_vec: {sent_vec.tolist()}")
        print(f"'{sentence}': {model.predict(sent_vec)}")  # 1: 负向


if __name__ == "__main__":
    start_time = time.time()
    preprocess_obj = Preprocessing()
    """
    preprocess_obj.get_sent_max_length()
    exit(0)
    sentence = "刘晓伟 好人"  # gen_sentence_vec()函数里"fasttext"的情况感觉也得处理成这种情况(空格分格)?
    # sentence = "刘晓伟好人"  # NOTE: 与空格分割得到的向量不同
    preprocess_obj.set_sent_vec_type("fasttext")
    print(f'fasttext: {preprocess_obj.gen_sentence_vec(sentence)}')
    preprocess_obj.set_sent_vec_type("avg")
    print(f'avg: {preprocess_obj.gen_sentence_vec(sentence)}')
    preprocess_obj.set_sent_vec_type("concatenate")
    print(f'concatenate: {preprocess_obj.gen_sentence_vec(sentence)}')
    exit(0)
    """

    sent_vec_type_list = ["avg", "fasttext", "concatenate"]
    sent_vec_type = sent_vec_type_list[2]
    print(f"\n{sent_vec_type} and", end=" ")
    preprocess_obj.set_sent_vec_type(sent_vec_type)

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    # print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    # print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    sent_analyse = SentimentAnalysis()
    algorithm_list = ["nb", "dt", "knn", "svm", "mlp", "cnn", "lstm"]
    algorithm_name = algorithm_list[3]
    print(f"{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name)
    model_cls = sent_analyse.get_model_class()
    model = sent_analyse.model_train(model_cls, X_train, y_train)
    sent_analyse.model_save(model)
    sent_analyse.model_evaluate(model, X_val, y_val)
    sent_analyse.model_predict(model, preprocess_obj)
    end_time = time.time()
    print(f"\nProgram Running Cost {end_time -start_time:.2f}s")

