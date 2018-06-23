# Sentiment Analysis Implementations
基于电影评论数据的中文情感分析  
**Chinese Sentiment Analysis** based on ML(Machine Learning) and DL(Deep Learning) algorithms, including Naive-Bayes, Decision-Tree, KNN, SVM, MLP, CNN, RNN(LSTM).  

## Requirements
All code in this project is implemented in [Python3.6+](https://www.python.org/downloads/).  
And all the essential packages are listed in `requirements.txt`, you can install them by `pip install -r requirements.txt -i https://pypi.douban.com/simple/`  
[Anaconda](https://docs.anaconda.com/anaconda/) or [virtualenv + virtualenvwrapper](http://www.jianshu.com/p/44ab75fbaef2) are strongly recommended to manage your Python environments.

## Data Preparation
**1.数据集**  
使用电影评论数据作为训练数据集. 其中, 训练数据集20000条(正负向各10000条); 测试数据集6000条(正负向各3000条)  

**2.数据预处理**  
1.去除停用词, 并使用jieba进行分词  
2.使用预训练的词向量模型，对句子进行向量化  

## 各种实现方法准确率对比
| Algorithm | Accuracy(avg) | Accuracy(fasttext) | Accuracy(matrix) |
| :---: | :---: | :---: | :---: |
| Naive-Bayes | 73.72% | 74.32% | 69.34% |
| Decision-Tree | 65.27% | 66.84% | 55.34% |
| KNN | 76.33% | 76.68% |  |
| SVM | 78.96% | 78.73% |  |
| MLP | 0.8359 |  |  |
| CNN | 0.8376 |  |  |
| LSTM | 0.8505 |  |  |
