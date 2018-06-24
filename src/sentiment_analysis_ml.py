#!/usr/bin/env python3
# coding: utf-8
# File: sentiment_analysis_ml.py
# Author: lxw
# Date: 12/21/17 8:13 AM

"""
基于"预训练词向量模型"和"机器学习"的情感分类(sklearn)
"""

import numpy as np
import pandas as pd
import time

from keras.preprocessing import sequence
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from preprocessing import Preprocessing


class SentimentAnalysis:
    def __init__(self, sent_vec_type):
        self.model_path_prefix="../data/output/models/"
        self.algorithm_name = "nb"
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}.model"

    def pick_algorithm(self, algorithm_name, sent_vec_type):
        assert algorithm_name in ["nb", "dt", "knn", "svm"], "algorithm_name must be in ['nb', 'dt', 'knn', 'svm']"
        self.algorithm_name = algorithm_name
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}..model"

    def model_build(self):
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
            """
            # OK
            model_cls = SVC(kernel="linear")
            tuned_parameters = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
                                {"kernel": ["linear"], "C": [1, 10, 100, 1000]}]
            model_cls = GridSearchCV(model_cls, tuned_parameters, cv=5, scoring="precision_weighted")
            """
            model_cls = SVC(C=1000, gamma=1e-3, kernel="rbf")  # avg
            # model_cls = SVC(C=1000, kernel="linear")  # fasttext

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
        # if self.algorithm_name in {"svm", "knn"}:
        #     print(model_cls.best_params_)
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

        sentence_df = pd.read_csv("../data/input/training_set.txt", sep="\t", header=None, names=["label", "sentence"])
        sentence_df = sentence_df.sample(frac=1)
        sentence_series = sentence_df["sentence"]
        label_series = sentence_df["label"]
        print(f"label_series: {label_series.iloc[:11]}")
        count = 0
        for sentence in sentence_series:
            count += 1
            sentence = sentence.strip()
            sent_vec = np.array(preprocess_obj.gen_sentence_vec(sentence)).reshape(1, -1)
            # print(f"sent_vec: {sent_vec.tolist()}")
            if preprocess_obj.sentence_vec_type == "concatenate":
                sent_vec = sequence.pad_sequences(sent_vec, maxlen=preprocess_obj.MAX_SENT_LEN * preprocess_obj.vector_size,
                                                  value=0, dtype=np.float)
                # print(f"sent_vec: {sent_vec.tolist()}")
            print(f"'{sentence}': {model.predict(sent_vec)}")  # 0: 正向, 1: 负向
            if count > 10:
                break


if __name__ == "__main__":
    start_time = time.time()
    preprocess_obj = Preprocessing()

    # 数据准备、模型训练、模型保存、模型评估、模型测试
    sent_vec_type_list = ["avg", "fasttext", "concatenate"]
    sent_vec_type = sent_vec_type_list[0]
    print(f"\n{sent_vec_type} and", end=" ")
    preprocess_obj.set_sent_vec_type(sent_vec_type)

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    # print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    # print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    sent_analyse = SentimentAnalysis(sent_vec_type)
    algorithm_list = ["nb", "dt", "knn", "svm"]
    algorithm_name = algorithm_list[3]
    print(f"{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name, sent_vec_type)
    model_cls = sent_analyse.model_build()
    model = sent_analyse.model_train(model_cls, X_train, y_train)
    sent_analyse.model_save(model)
    sent_analyse.model_evaluate(model, X_val, y_val)
    sent_analyse.model_predict(model, preprocess_obj)
    end_time = time.time()
    print(f"\nProgram Running Cost {end_time -start_time:.2f}s")

    """
    # 模型导入、模型测试
    sent_analyse = SentimentAnalysis(sent_vec_type)
    algorithm_list = ["nb", "dt", "knn", "svm"]
    algorithm_name = algorithm_list[0]
    print(f"{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name, sent_vec_type)
    model = joblib.load(sent_analyse.model_path)

    sent_analyse.model_predict(model, preprocess_obj)
    end_time = time.time()
    print(f"\nProgram Running Cost {end_time -start_time:.2f}s")
    """

