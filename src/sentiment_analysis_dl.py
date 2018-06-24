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
import pickle
import time

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

from preprocessing import Preprocessing


class SentimentAnalysis:
    def __init__(self, preprocess_obj, sent_vec_type):
        self.model_path_prefix="../data/output/models/"
        self.algorithm_name = "nn"
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}"
        self.preprocess_obj = preprocess_obj
        self.sent_vec_type = sent_vec_type
        self.bath_size = 512  # TODO
        self.epochs = 1000  # TODO

    def pick_algorithm(self, algorithm_name, sent_vec_type):
        assert algorithm_name in ["nn", "cnn", "lstm"], "algorithm_name must be in ['nn', 'cnn', 'lstm']"
        self.algorithm_name = algorithm_name
        self.model_path = f"{self.model_path_prefix}{self.algorithm_name}_{sent_vec_type}"
        self.sent_vec_type = sent_vec_type

    def model_build(self, input_shape):
        model_cls = Sequential()
        if self.algorithm_name == "nn":  # Neural Network(Multi-Layer Perceptron)
            # activation is essential for Dense, otherwise linear is used.
            model_cls.add(Dense(64, input_shape=input_shape, activation="relu", name="dense1"))
            model_cls.add(Dropout(0.25, name="dropout2"))
            model_cls.add(Dense(64, activation="relu", name="dense3"))
            model_cls.add(Dropout(0.25, name="dropout4"))
            model_cls.add(Dense(2, activation="softmax", name="dense5"))
            # model_cls.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # TODO:
            model_cls.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        elif self.algorithm_name == "cnn":
            # input_shape = (rows行, cols列, 1) 1表示颜色通道数目, rows行，对应一句话的长度, cols列表示词向量的维度
            model_cls.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))  # filters, kernel_size
            model_cls.add(Conv1D(64, 3, activation="relu"))
            model_cls.add(MaxPooling1D(3))
            model_cls.add(Conv1D(128, 3, activation="relu"))
            model_cls.add(Conv1D(128, 3, activation="relu"))
            model_cls.add(GlobalAveragePooling1D())
            model_cls.add(Dropout(0.25))
            model_cls.add(Dense(2, activation="sigmoid"))

            model_cls.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # TODO: categorical_crossentropy
        elif self.algorithm_name == "lstm":
            model_cls.add(Masking(mask_value=-1, input_shape=input_shape, name="masking_layer"))
            model_cls.add(LSTM(units=64, return_sequences=True, dropout=0.25, name="lstm1"))
            model_cls.add(LSTM(units=128, return_sequences=False, dropout=0.25, name="lstm2"))
            model_cls.add(Dense(units=2, activation="softmax", name="dense5"))

            model_cls.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

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
        # model_path = f"../data/output/models/{self.algorithm_name}_best_model_{epoch:02d}_{val_loss:.2f}.hdf5"  # NO
        model_path = "../data/output/models/best_model_{epoch:02d}_{val_loss:.2f}.hdf5"  # OK
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True,
                                     mode="min")

        hist_obj = model_cls.fit(X_train, y_train, batch_size=self.bath_size, epochs=self.epochs, verbose=1,
                                 validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reduction, checkpoint])
        with open(f"../data/output/history_{self.algorithm_name}_{self.sent_vec_type}.pkl", "wb") as f:
            pickle.dump(hist_obj.history, f)
        return model_cls  # model

    def plot_hist(self, history_filename):
        import matplotlib.pyplot as plt

        history = None
        with open(f"../data/output/{history_filename}.pkl", "rb") as f:
            history = pickle.load(f)

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
        print("model.metrics:{0}, model.metrics_names:{1}".format(model.metrics, model.metrics_names))
        scores = model.evaluate(X_val, y_val)
        loss, accuracy = scores[0], scores[1] * 100
        print(f"Loss: {loss:.2f}, '{self.algorithm_name}' Classification Accuracy: {accuracy:.2f}%")

    def model_predict(self, model):
        """
        模型测试
        :param model: 训练好的模型对象
        :param preprocess_obj: Preprocessing类对象
        :return: None
        """
        sentence = "这件 衣服 真的 太 好看 了 ！ 好想 买 啊 "  # TODO:
        sent_vec = np.array(self.preprocess_obj.gen_sentence_vec(sentence))  # shape: (70, 200)
        if self.sent_vec_type == "matrix":  # cnn or lstm
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0], sent_vec.shape[1])
        elif self.algorithm_name == "nn":  # nn.
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0])
        elif self.algorithm_name == "lstm" and (self.sent_vec_type == "avg" or self.sent_vec_type == "fasttext"):  # lstm.
            sent_vec = sent_vec.reshape(1, 1, sent_vec.shape[0])
        print(f"'{sentence}': {np.argmax(model.predict(sent_vec))}")  # 0: 正向

        sentence = "这 真的是 一部 非常 优秀 电影 作品"
        sent_vec = np.array(self.preprocess_obj.gen_sentence_vec(sentence))
        if self.sent_vec_type == "matrix":  # cnn or lstm
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0], sent_vec.shape[1])
        elif self.algorithm_name == "nn":  # nn.
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0])
        elif self.algorithm_name == "lstm" and (self.sent_vec_type == "avg" or self.sent_vec_type == "fasttext"):  # lstm.
            sent_vec = sent_vec.reshape(1, 1, sent_vec.shape[0])
        print(f"'{sentence}': {np.argmax(model.predict(sent_vec))}")  # 0: 正向

        sentence = "这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了"
        sent_vec = np.array(self.preprocess_obj.gen_sentence_vec(sentence))
        if self.sent_vec_type == "matrix":  # cnn or lstm
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0], sent_vec.shape[1])
        elif self.algorithm_name == "nn":  # nn.
            sent_vec = sent_vec.reshape(1, sent_vec.shape[0])
        elif self.algorithm_name == "lstm" and (self.sent_vec_type == "avg" or self.sent_vec_type == "fasttext"):  # lstm.
            sent_vec = sent_vec.reshape(1, 1, sent_vec.shape[0])
        print(f"'{sentence}': {np.argmax(model.predict(sent_vec))}")  # 1: 负向

        sentence_df = pd.read_csv("../data/input/training_set.txt", sep="\t", header=None, names=["label", "sentence"])
        sentence_df = sentence_df.sample(frac=1)
        sentence_series = sentence_df["sentence"]
        label_series = sentence_df["label"]
        print(f"label_series: {label_series.iloc[:11]}")
        count = 0
        for sentence in sentence_series:
            count += 1
            sentence = sentence.strip()
            sent_vec = np.array(self.preprocess_obj.gen_sentence_vec(sentence))
            if self.sent_vec_type == "matrix":  # cnn or lstm
                sent_vec = sent_vec.reshape(1, sent_vec.shape[0], sent_vec.shape[1])
            elif self.algorithm_name == "nn":  # nn.
                sent_vec = sent_vec.reshape(1, sent_vec.shape[0])
            elif self.algorithm_name == "lstm" and (self.sent_vec_type == "avg" or self.sent_vec_type == "fasttext"):  # lstm.
                sent_vec = sent_vec.reshape(1, 1, sent_vec.shape[0])
            print(f"'{sentence}': {np.argmax(model.predict(sent_vec))}")  # 0: 正向, 1: 负向
            if count > 10:
                break


if __name__ == "__main__":
    start_time = time.time()
    preprocess_obj = Preprocessing()
    '''

    sent_vec_type_list = ["avg", "fasttext", "matrix"]  # NN(MLP): 只能使用avg或fasttext. CNN: 只能使用matrix. LSTM: avg, fasttext, matrix均可.
    sent_vec_type = sent_vec_type_list[0]
    print(f"\n{sent_vec_type} and", end=" ")
    preprocess_obj.set_sent_vec_type(sent_vec_type)

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)

    sent_analyse = SentimentAnalysis(preprocess_obj, sent_vec_type)
    algorithm_list = ["nn", "cnn", "lstm"]
    algorithm_name = algorithm_list[0]

    if len(X_train.shape) == 2 and algorithm_name == "lstm":  # avg or fasttext
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        input_shape = (X_train.shape[1], X_train.shape[2])
    elif algorithm_name == "nn":
        input_shape = (X_train.shape[1],)
    else:  # algorithm_name == "cnn" or (algorithm_name == "lstm" and len(X_train.shape) == 3)
        input_shape = (X_train.shape[1], X_train.shape[2])
    print(X_train.shape, y_train.shape)  # (19998, 200)/(19998, 70, 200) (19998,)
    print(X_val.shape, y_val.shape)  # (5998, 200)/(5998, 70, 200) (5998,)

    print(f"{algorithm_name}:")
    sent_analyse.pick_algorithm(algorithm_name, sent_vec_type)
    # """
    model_cls = sent_analyse.model_build(input_shape=input_shape)
    model = sent_analyse.model_train(model_cls, X_train, X_val, y_train, y_val)
    # """
    sent_analyse.model_evaluate(model, X_val, y_val)
    sent_analyse.model_predict(model)
    '''
    sent_analyse = SentimentAnalysis(preprocess_obj, "")
    sent_analyse.plot_hist("history_nn_fasttext")
    end_time = time.time()
    print(f"\nProgram Running Cost {end_time -start_time:.2f}s")
