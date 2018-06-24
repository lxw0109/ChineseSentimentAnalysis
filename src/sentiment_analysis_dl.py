#!/usr/bin/env python3
# coding: utf-8
# File: sentiment_analysis_dl.py
# Author: lxw
# Date: 12/21/17 9:23 AM

"""
基于"预训练词向量模型"和"深度学习"的情感分类(keras)
"""

import numpy as np
import pandas as pd
import time

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.preprocessing import sequence

from .preprocessing import Preprocessing


class SentimentAnalysis:
    def __init__(self, preprocess_obj, sent_vec_type):
        self.model_path_prefix="../data/output/models/"
        self.algorithm_name = "nn"
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}"
        self.preprocess_obj = preprocess_obj
        self.bath_size = 32
        self.epochs = 1000

    def pick_algorithm(self, algorithm_name, sent_vec_type):
        assert algorithm_name in ["nn", "cnn", "lstm"], "algorithm_name must be in ['nn', 'cnn', 'lstm']"
        self.algorithm_name = algorithm_name
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}"

    def model_build(self):
        model_cls = Sequential()
        if self.algorithm_name == "nn":  # Neural Network(Multi-Layer Perceptron)
            # activation is essential for Dense, otherwise linear is used.
            model_cls.add(Dense(64, input_shape=(self.preprocess_obj.vector_size, self.preprocess_obj.MAX_SENT_LEN),
                                activation="relu"))
            model_cls.add(Dropout(0.25))
            model_cls.add(Dense(64, activation="relu"))
            model_cls.add(Dropout(0.25))
            model_cls.add(Dense(1, activation="sigmoid"))
            model_cls.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        elif self.algorithm_name == "cnn":
            pass
        elif self.algorithm_name == "lstm":
            pass

        return model_cls

    def model_train(self, model_cls, X_train, X_val, y_train, y_val):
        """
        分类器模型的训练
        :param model_cls: 所使用的算法的类的定义，尚未训练的模型
        :param X_train: 
        :param y_train: 
        :param X_val: 
        :param y_val: 
        :return: 训练好的模型
        """
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)
        lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, factor=0.2, min_lr=1e-5)
        # 检查最好模型: 只要有提升, 就保存一次. 保存到多个模型文件
        model_path = f"../data/output/models/{self.algorithm_name}_best_model_{epoch:02d}_{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True,
                                     mode="min")

        hist_obj = model_cls.fit(X_train, y_train, batch_size=self.bath_size, epochs=self.epochs, verbose=1,
                             validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reduction, checkpoint])
        self.plot_hist(hist_obj.history)
        return model_cls  # model

    def plot_hist(self, history):
        import matplotlib.pyplot as plt

        if not history:
            return
        # 绘制训练集和验证集的曲线
        plt.plot(history["acc"], label="Training Accuracy", color="green", linewidth=1)
        plt.plot(history["loss"], label="Training Loss", color="red", linewidth=1)
        plt.plot(history["val_acc"], label="Validation Accuracy", color="purple", linewidth=1)
        plt.plot(history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
        plt.grid(True)  # 设置网格形式
        plt.xlabel("epoch")
        plt.ylabel("acc-loss")  # 给x, y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.show()

    def model_evaluate(self, model, X_val, y_val):
        """
        分类器模型的评估
        :param model: 训练好的模型对象
        :param X_val: 
        :param y_val: 
        :return: None
        """
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

        # Model Evaluation
        print("model.metrics:{0}, model.metrics_names:{1}".format(model.metrics, model.metrics_names))
        scores = model.evaluate(X_val, y_val)
        loss, accuracy = scores[0], scores[1] * 100
        print("Loss: {0:.2f}, Model Accuracy: {1:.2f}%".format(loss, accuracy))

    def model_predict(self, model, preprocess_obj):
        """
        模型测试
        :param model: 训练好的模型对象
        :param preprocess_obj: Preprocessing类对象
        :return: None
        """
        sentence = "这 真的是 一部 非常 优秀 电影 作品"
        sent_vec = np.array(preprocess_obj.gen_sentence_vec(sentence))  # shape: (1, 1000)
        print(f"'{sentence}': {model.predict(sent_vec)}")  # 0: 正向

        sentence = "这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了"
        sent_vec = np.array(preprocess_obj.gen_sentence_vec(sentence))
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
            print(f"'{sentence}': {model.predict(sent_vec)}")  # 0: 正向, 1: 负向
            if count > 10:
                break


if __name__ == "__main__":
    start_time = time.time()
    preprocess_obj = Preprocessing()

    sent_vec_type_list = ["avg", "fasttext", "concatenate"]
    sent_vec_type = sent_vec_type_list[0]
    print(f"\n{sent_vec_type} and", end=" ")
    preprocess_obj.set_sent_vec_type(sent_vec_type)

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    # print(X_train.shape, y_train.shape)  # (19998, 100) (19998,)
    # print(X_val.shape, y_val.shape)  # (5998, 100) (5998,)

    sent_analyse = SentimentAnalysis(sent_vec_type)
    algorithm_list = ["nb", "dt", "knn", "svm"]
    algorithm_name = algorithm_list[0]
    print(f"{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name, sent_vec_type)
    # """
    model_cls = sent_analyse.model_build()
    model = sent_analyse.model_train(model_cls, X_train, y_train)
    # """
    sent_analyse.model_evaluate(model, X_val, y_val)
    sent_analyse.model_predict(model, preprocess_obj)
    end_time = time.time()
    print(f"\nProgram Running Cost {end_time -start_time:.2f}s")
