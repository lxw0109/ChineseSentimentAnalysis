# Sentiment Analysis Implementations
**Chinese Sentiment Analysis** based on ML(Machine Learning) and DL(Deep Learning) algorithms, such as Naive-Bayes, SVM, Decision-Tree, KNN, MLP, CNN, RNN(LSTM).

## Requirements
All code in this project is implemented in [Python3.6+](https://www.python.org/downloads/).  
And all the essential packages are listed in `requirements.txt`, you can install them by `pip install -r requirements.txt -i https://pypi.douban.com/simple/`  
[Anaconda](https://docs.anaconda.com/anaconda/) or [virtualenv + virtualenvwrapper](http://www.jianshu.com/p/44ab75fbaef2) are strongly recommended to manage your Python environments.

## Data Preparation
**1.数据集**  
使用电影评论数据作为训练数据集  
训练数据集20000(正向10000, 负向10000)  
测试数据集6000条(正向3000, 负向3000)  

**2.数据处理**  
1.使用jieba进行分词(**去除了标点符号，停用词好像没有去除?**)  
2.使用预训练的词向量模型，对句子进行向量化  

## 训练与对比(准确率)
| Algorithm | Accuracy |
| :---: | :---: |
| Naive-Bayes | {avg: 73.72%, fasttext: 74.32%, matrix: %} |
| Decision-Tree | 0.6907 |
| KNN | 0.7909 |
| SVM | 0.8303 |
| MLP | 0.8359 |
| CNN | 0.8376 |
| LSTM | 0.8505 |
