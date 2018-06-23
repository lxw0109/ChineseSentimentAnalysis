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

# from collections import Counter
from keras.preprocessing import sequence
from pyfasttext import FastText
from sklearn.externals import joblib


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

        self.MAX_SENT_LEN = 36  # DEBUG: 超参数. self.get_sent_max_length()
        # TODO: using gridsearchcv.
        # 100: 50.22%, 80: 50.23%, 60: 55.92%, 50: 69.11%, 40: 68.91%, 36: 69.34%, 30: 69.22%, 20: 69.17%, 10: 67.07%

    def set_sent_vec_type(self, sentence_vec_type):
        assert sentence_vec_type in ["avg", "concatenate", "fasttext"], \
            "sentence_vec_type must be in ['avg', 'fasttext', 'concatenate']"
        self.sentence_vec_type = sentence_vec_type

    def get_sent_max_length(self):  # NOT_USED
        # sent_len_counter = Counter()
        max_length = 0
        with open("../data/input/training_set.txt") as f:
            for line in f:
                content = line.strip().split("\t")[1]
                length = len(content)
                # sent_len_counter[length] += 1
                if max_length <= length:  # TODO: 训练数据中好像没有这么长的文本吧
                    max_length = length
        # sent_len_counter = sorted(list(sent_len_counter.items()), key=lambda x: x[0])
        # print(sent_len_counter)
        # [(63, 1), (65, 2), (72, 1), (73, 2), (74, 2), (75, 9), (76, 8), (77, 12), (78, 20), (79, 19), (80, 42), (81, 76), (82, 86), (83, 132), (84, 153), (85, 189), (86, 235), (87, 250), (88, 281), (89, 282), (90, 315), (91, 326), (92, 307), (93, 316), (94, 342), (95, 309), (96, 310), (97, 300), (98, 308), (99, 288), (100, 309), (101, 264), (102, 291), (103, 256), (104, 280), (105, 263), (106, 258), (107, 274), (108, 273), (109, 232), (110, 245), (111, 234), (112, 203), (113, 246), (114, 215), (115, 224), (116, 209), (117, 196), (118, 226), (119, 199), (120, 198), (121, 178), (122, 199), (123, 165), (124, 172), (125, 218), (126, 170), (127, 163), (128, 173), (129, 161), (130, 198), (131, 163), (132, 153), (133, 173), (134, 173), (135, 156), (136, 141), (137, 148), (138, 186), (139, 168), (140, 157), (141, 158), (142, 147), (143, 126), (144, 141), (145, 173), (146, 154), (147, 153), (148, 162), (149, 172), (150, 181), (151, 153), (152, 165), (153, 176), (154, 191), (155, 180), (156, 177), (157, 172), (158, 194), (159, 193), (160, 216), (161, 155), (162, 188), (163, 171), (164, 181), (165, 187), (166, 181), (167, 190), (168, 176), (169, 171), (170, 160), (171, 166), (172, 152), (173, 119), (174, 111), (175, 115), (176, 96), (177, 74), (178, 79), (179, 57), (180, 49), (181, 52), (182, 26), (183, 20), (184, 27), (185, 25), (186, 21), (187, 25), (188, 10), (189, 12), (190, 15), (191, 8), (192, 4), (193, 11), (194, 5), (195, 10), (196, 3), (197, 1), (198, 2), (199, 3), (200, 3), (201, 1), (202, 1), (203, 1), (204, 2), (205, 2), (206, 3), (208, 1), (209, 2), (210, 3), (211, 2), (212, 1), (214, 2), (216, 1), (217, 1), (218, 2), (219, 1), (220, 1), (222, 1), (226, 1), (228, 2), (231, 2), (234, 2), (235, 2), (236, 1), (237, 1), (238, 1), (239, 1), (241, 3), (243, 1), (244, 1), (245, 1), (246, 4), (247, 3), (251, 1), (253, 1), (254, 1), (255, 1), (257, 1), (258, 1), (262, 1), (265, 2), (266, 1), (268, 2), (271, 1), (272, 1), (273, 3), (274, 1), (276, 1), (277, 2), (278, 1), (280, 1), (281, 1), (282, 2), (284, 2), (286, 1), (287, 1), (288, 3), (289, 2), (290, 1), (293, 1), (295, 1), (297, 1), (298, 2), (299, 2), (300, 1), (301, 1), (302, 2), (304, 2), (305, 1), (307, 1), (308, 1), (309, 2), (311, 1), (312, 1), (313, 2), (314, 1), (317, 1), (319, 1), (321, 1), (322, 1), (326, 1), (329, 1), (1954, 1)]
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
        # print(result)
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
    sent_vec_type = sent_vec_type_list[1]
    print(f"\n{sent_vec_type}:")
    preprocess_obj.set_sent_vec_type(sent_vec_type)

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    # print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    # print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    sent_analyse = SentimentAnalysis()
    algorithm_list = ["nb", "dt", "knn", "svm", "mlp", "cnn", "lstm"]
    algorithm_name = algorithm_list[1]
    print(f"\n{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name)
    model_cls = sent_analyse.get_model_class()
    model = sent_analyse.model_train(model_cls, X_train, y_train)
    sent_analyse.model_save(model)
    sent_analyse.model_evaluate(model, X_val, y_val)
    sent_analyse.model_predict(model, preprocess_obj)
    end_time = time.time()
    print(f"Program Running Cost {end_time -start_time:.2f}s")

