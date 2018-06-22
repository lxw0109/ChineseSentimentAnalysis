#!/usr/bin/env python3
# coding: utf-8
# File: naive_bayes_imp.py
# Author: lxw
# Date: 17-12-20

"""
基于 fastText预训练词向量模型 和 Naive-Bayes 的情感分类
"""

import numpy as np

from pyfasttext import FastText
from sklearn.externals import joblib


class NB_Implement():
    def __init__(self):
        self.model = FastText("../data/input/models/sg_pyfasttext.bin")  # DEBUG
        # self.model = FastText("../data/input/models/880w_fasttext_skip_gram.bin")

        # self.vocab_size, self.vector_size = self.model.numpy_normalized_vectors.shape  # OK
        self.vocab_size = self.model.nwords
        self.vector_size = self.model.args.get("dim")


    def gen_sentence_vec(self, sentence, sentence_vec_type="matrix"):
        """
        :param sentence: 
        :param sentence_vec_type:
          句子的表示形式: {"avg": 向量和的平均, "fasttext": get_numpy_sentence_vector, "matrix": matrix}
        :return: 
        """
        sentence = sentence.strip()
        if sentence_vec_type == "fasttext":
            return self.model.get_numpy_sentence_vector(sentence)

        word_list = [word for word in sentence.split(" ")]
        word_len = len(word_list)
        if sentence_vec_type == "matrix":
            sentence_matrix = np.empty(word_len, dtype=list)
            for idx, word in enumerate(word_list):
                sentence_matrix[idx] = self.model.get_numpy_vector(word)
            return sentence_matrix
        else:  # sentence_vec_type == "avg":
            assert sentence_vec_type == "avg", "sentence_vec_type must be in (avg, fasttext, matrix)"
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
        X_val = list()
        y_val = list()
        for line in open("../data/input/training_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1], "avg")
            X_train.append(sent_vector)

            if line[0] == "1":
                y_train.append(1)
            else:
                y_train.append(0)

        for line in open("../data/input/validation_set.txt"):
            line = line.strip().split("\t")
            sent_vector = self.gen_sentence_vec(line[-1], "avg")
            X_val.append(sent_vector)
            if line[0] == "1":
                y_val.append(1)
            else:
                y_val.append(0)

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val),


    def train_bayes(self, X_train, y_train):
        """
        基于Naive Bayes的分类算法
        """
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X_train, y_train)
        joblib.dump(model, "../data/output/models/bayes.model")

    def evaluate_bayes(self, model_path, X_val, y_val):
        """
        基于Naive Bayes分类器的预测
        """
        model = joblib.load(model_path)
        y_predict = list()
        y_val = list(y_val)
        correct = 0
        for sent_vec in X_val:  # sent_vec.shape: (self.vector_size,)
            predicted = model.predict(sent_vec.reshape(1, -1))  # sent_vec.reshape(1, -1).shape: (1, self.vector_size)
            y_predict.append(predicted[0])
        for idx in range(len(y_predict)):
            if int(y_predict[idx]) == int(y_val[idx]):
                correct += 1
        score = correct / len(y_predict)
        print(f"Bayes Classification Accuray:{score}")  # 56.05%
        return score


    def predict_bayes(self, model_path):
        """
        实际应用测试
        """
        model = joblib.load(model_path)
        sentence1 = "这件 衣服 真的 太 好看 了 ！ 好想 买 啊 "
        sentence2 = "这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了"
        sent_vec1 = np.array(self.gen_sentence_vec(sentence1, "avg")).reshape(1, -1)
        sent_vec2 = np.array(self.gen_sentence_vec(sentence2, "avg")).reshape(1, -1)
        print(f"'{sentence1}': {model.predict(sent_vec1)}")  # 0: 正向
        print(f"'{sentence2}': {model.predict(sent_vec2)}")  # 1: 负向


if __name__ == "__main__":
    nb = NB_Implement()

    """
    sentence = "刘晓伟 好人"  # gen_sentence_vec()函数里"fasttext"的情况感觉也得处理成这种情况(空格分格)?
    # sentence = "刘晓伟好人"  # 与空格分割得到的向量不同
    print(f'fasttext: {nb.gen_sentence_vec(sentence, sentence_vec_type="fasttext")}')
    print(f'avg: {nb.gen_sentence_vec(sentence, sentence_vec_type="avg")}')
    print(f'matrix: {nb.gen_sentence_vec(sentence, sentence_vec_type="matrix")}')
    """

    X_train, y_train, X_val, y_val = nb.gen_train_val_data()
    print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    nb.train_bayes(X_train, y_train)

    model_path = "../data/output/models/bayes.model"
    nb.evaluate_bayes(model_path, X_val, y_val)
    nb.predict_bayes(model_path)
