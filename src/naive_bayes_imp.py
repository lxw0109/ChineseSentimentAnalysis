#!/usr/bin/env python3
# coding: utf-8
# File: naive_bayes_imp.py
# Author: lxw
# Date: 17-12-20

"""
基于 fastText预训练词向量模型 和 Naive-Bayes 的情感分类
"""

import numpy as np
import time

from pyfasttext import FastText
from sklearn.externals import joblib


class NB_Implement():
    def __init__(self):
        start_time = time.time()
        # self.model = FastText("../data/input/models/sg_pyfasttext.bin")  # DEBUG
        self.model = FastText("../data/input/models/880w_fasttext_skip_gram.bin")
        end_time = time.time()
        print(f"Loading word vector model cost: {end_time - start_time:.2f}s")

        # self.vocab_size, self.vector_size = self.model.numpy_normalized_vectors.shape  # OK
        self.vocab_size = self.model.nwords
        self.vector_size = self.model.args.get("dim")
        print(f"self.vector_size:{self.vector_size}, self.vocab_size: {self.vocab_size}")  # self.vector_size:200, self.vocab_size: 925242

        # 句子的表示形式: {"avg": 向量和的平均, "fasttext": get_numpy_sentence_vector, "matrix": matrix}
        self.sentence_vec_type = "avg"

    def set_sent_vec_type(self, sentence_vec_type):
        assert self.sentence_vec_type in ["avg", "matrix", "fasttext"], "self.sentence_vec_type must be in ['avg', 'fasttext', 'matrix']"
        self.sentence_vec_type = sentence_vec_type

    def gen_sentence_vec(self, sentence):
        """
        :param sentence: 
        :return: 
        """
        sentence = sentence.strip()
        if self.sentence_vec_type == "fasttext":
            return self.model.get_numpy_sentence_vector(sentence)

        word_list = [word for word in sentence.split(" ")]
        word_len = len(word_list)
        if self.sentence_vec_type == "matrix":
            sentence_matrix = np.empty(word_len, dtype=list)
            for idx, word in enumerate(word_list):
                sentence_matrix[idx] = self.model.get_numpy_vector(word)
            return sentence_matrix
        else:  # self.sentence_vec_type == "avg":
            sentence_vector = np.zeros(self.vector_size)  # <ndarray>
            # print(f"type(sentence_vector): {type(sentence_vector)}")
            for idx, word in enumerate(word_list):
                # print(f"type(self.model.get_numpy_vector(word)): {type(self.model.get_numpy_vector(word))}")  # <ndarry>
                sentence_vector += self.model.get_numpy_vector(word)
            return sentence_vector / len(word_list)


    def gen_train_val_data(self):
        """
        构造训练, 验证数据
        """
        X_train = list()
        y_train = list()
        for line in open("../data/input/training_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1])
            X_train.append(sent_vector)
            y_train.append(int(line[0]))

        X_val = list()
        y_val = list()
        for line in open("../data/input/validation_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1])
            X_val.append(sent_vector)
            y_val.append(int(line[0]))

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val),


    def train_bayes(self, X_train, y_train):
        """
        基于Naive Bayes的分类算法
        """
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X_train, y_train)
        joblib.dump(model, "../data/output/models/bayes_model")


    def evaluate_bayes(self, model_path, X_val, y_val):
        """
        基于Naive Bayes分类器的预测
        """
        model = joblib.load(model_path)
        y_val = list(y_val)
        correct = 0
        """
        y_predict = list()
        for sent_vec in X_val:  # sent_vec.shape: (self.vector_size,)
            predicted = model.predict(sent_vec.reshape(1, -1))  # sent_vec.reshape(1, -1).shape: (1, self.vector_size)
            y_predict.append(predicted[0])
        """
        y_predict = model.predict(X_val)
        print(f"len(y_predict): {len(y_predict)}, len(y_val): {len(y_val)}")
        assert len(y_predict) == len(y_val), "Unexpected Error: len(y_predict) != len(y_val), but it should be"
        for idx in range(len(y_predict)):
            if int(y_predict[idx]) == int(y_val[idx]):
                correct += 1
        score = correct / len(y_predict)
        print(f"Bayes Classification Accuray:{score}")
        return score


    def predict_bayes(self, model_path):
        """
        实际应用测试
        """
        model = joblib.load(model_path)
        sentence = "这件 衣服 真的 太 好看 了 ！ 好想 买 啊 "
        sent_vec = np.array(self.gen_sentence_vec(sentence)).reshape(1, -1)
        print(f"'{sentence}': {model.predict(sent_vec)}")  # 1: 负向

        sentence = "这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了"
        sent_vec = np.array(self.gen_sentence_vec(sentence)).reshape(1, -1)
        print(f"'{sentence}': {model.predict(sent_vec)}")  # 1: 负向


if __name__ == "__main__":
    nb = NB_Implement()

    """
    sentence = "刘晓伟 好人"  # gen_sentence_vec()函数里"fasttext"的情况感觉也得处理成这种情况(空格分格)?
    # sentence = "刘晓伟好人"  # 与空格分割得到的向量不同
    nb.set_sent_vec_type("fasttext")
    print(f'fasttext: {nb.gen_sentence_vec(sentence)}')
    nb.set_sent_vec_type("avg")
    print(f'avg: {nb.gen_sentence_vec(sentence)}')
    nb.set_sent_vec_type("matrix")
    print(f'matrix: {nb.gen_sentence_vec(sentence)}')
    """

    # 1. avg
    nb.set_sent_vec_type("avg")
    X_train, y_train, X_val, y_val = nb.gen_train_val_data()
    # print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    # print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    nb.train_bayes(X_train, y_train)

    model_path = "../data/output/models/bayes_model"
    nb.evaluate_bayes(model_path, X_val, y_val)  # 73.72%
    nb.predict_bayes(model_path)
    print("--" * 30)

    # 2. fasttext
    nb.set_sent_vec_type("fasttext")
    X_train, y_train, X_val, y_val = nb.gen_train_val_data()

    nb.train_bayes(X_train, y_train)

    model_path = "../data/output/models/bayes_model"
    nb.evaluate_bayes(model_path, X_val, y_val)  # 74.32%
    nb.predict_bayes(model_path)
    print("--" * 30)

    """
    # 3. matrix
    nb.set_sent_vec_type("matrix")
    X_train, y_train, X_val, y_val = nb.gen_train_val_data()

    nb.train_bayes(X_train, y_train)

    model_path = "../data/output/models/bayes_model"
    nb.evaluate_bayes(model_path, X_val, y_val)
    nb.predict_bayes(model_path)
    """
