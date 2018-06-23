#!/usr/bin/env python3
# coding: utf-8
# File: preprocessing.py
# Author: lxw
# Date: 12/20/17 8:10 AM


import numpy as np
import time

from keras.preprocessing import sequence
from collections import Counter
from pyfasttext import FastText


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


if __name__ == "__main__":
    preprocess_obj = Preprocessing()
    """
    preprocess_obj.get_sent_max_length()
    sentence = "刘晓伟 好人"  # gen_sentence_vec()函数里"fasttext"的情况感觉也得处理成这种情况(空格分格)?
    # sentence = "刘晓伟好人"  # NOTE: 与空格分割得到的向量不同
    preprocess_obj.set_sent_vec_type("fasttext")
    print(f'fasttext: {preprocess_obj.gen_sentence_vec(sentence)}')
    preprocess_obj.set_sent_vec_type("avg")
    print(f'avg: {preprocess_obj.gen_sentence_vec(sentence)}')
    preprocess_obj.set_sent_vec_type("concatenate")
    print(f'concatenate: {preprocess_obj.gen_sentence_vec(sentence)}')
    """

    X_train, X_val, y_train, y_val = preprocess_obj.gen_train_val_data()
    # print(f"X_train: {X_train}\nX_val: {X_val}\ny_train: {y_train}\ny_val: {y_val}")
    print(f"X_train.shape: {X_train.shape}\nX_val.shape: {X_val.shape}\n"
          f"y_train.shape: {y_train.shape}\ny_val.shape: {y_val.shape}")
