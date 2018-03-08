#encoding=utf8
import os
import pickle
from itertools import chain
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence import Sentence
from sentence import TagPrefix
from sentence import TagSurfix
'''
数据的预处理
'''

class DataHandler(object):
    def __init__(self, rootDir='raw_data', save_path='data/data.pkl'):
        self.rootDir = rootDir
        self.save_path = save_path
        self.spiltChar = ['。', '!', '！', '？', '?']
        self.max_len = 200
        self.totalLine = 0
        self.longLine = 0
        self.totalChars = 0
        self.TAGPRE = TagPrefix.convert()

    # 判断数据是否预处理
    def loadData(self):
        isFile = os.path.isfile(self.save_path)
        # 导入数据
        if isFile:
            with open(self.save_path, 'rb') as inp:
                self.X = pickle.load(inp)
                self.y = pickle.load(inp)
                self.word2id = pickle.load(inp)
                self.id2word = pickle.load(inp)
                self.tag2id = pickle.load(inp)
                self.id2tag = pickle.load(inp)
        else:
            self.loadRawData()
            self.handlerRawData()

    # 顺序读取文件，按行处理
    def loadRawData(self):
        self.datas = list()
        self.labels = list()
        if self.rootDir:
            print(self.rootDir)
            for dirName, subdirList, fileList in os.walk(self.rootDir):
                # curDir = os.path.join(self.rootDir, dirName)
                for file in fileList:                   
                    if file.endswith(".txt"): 
                        curFile = os.path.join(dirName, file)
                        print("processing:%s" % (curFile))
                        with open(curFile, "r", encoding='utf-8') as fp:
                            for line in fp.readlines():
                                self.processLine(line)

            print("total:%d, long lines:%d, total chars:%d" % (self.totalLine, self.longLine, self.totalChars))
            print('Length of datas is %d' % len(self.datas))
            print('Example of datas: ', self.datas[0])
            print('Example of labels:', self.labels[0])

    # 处理一行记录
    def processLine(self, line):
        line = line.strip()  # 去除掉前后空格
        nn = len(line)
        seeLeftB = False  # [纽约/nsf 时报/n]/nz 这种类型数据的开始标记，当为True时，代表后续的是这样的数据类型
        start = 0
        sentence = Sentence()  # 代表一个句子
        try:
            for i in range(nn):  # 循环遍历整个句子
                if line[i] == ' ':  # 一个token的结束
                    if not seeLeftB:
                        token = line[start:i]
                        if token.startswith('['):
                            tokenLen = len(token)
                            while tokenLen > 0 and token[tokenLen - 1] != ']':
                                tokenLen = tokenLen - 1
                            token = token[1:tokenLen - 1]
                            ss = token.split(' ')
                            for s in ss:  # 把[纽约/nsf 时报/n]/nz 中的两个子token分别做处理
                                self.processToken(s, sentence, False)
                        else:
                            self.processToken(token, sentence, False)
                        start = i + 1
                elif line[i] == '[':
                    seeLeftB = True
                elif line[i] == ']':
                    seeLeftB = False

            if start < nn:  # 该句子中最后一个token
                token = line[start:]
                if token.startswith('['):
                    tokenLen = len(token)
                    while tokenLen > 0 and token[tokenLen - 1] != ']':
                        tokenLen = tokenLen - 1
                    token = token[1:tokenLen - 1]
                    ss = token.split(' ')
                    ns = len(ss)
                    for i in range(ns - 1):
                        self.processToken(ss[i], sentence, False)
                    self.processToken(ss[-1], sentence, True)
                else:
                    self.processToken(token, sentence, True)
        except Exception as e:
            print('处理数据异常, 异常行为：' + line)
            print(e)

    # 代表处理一个单词、一个token
    def processToken(self, tokens, sentence, end):
        nn = len(tokens)
        while nn > 0 and tokens[nn - 1] != '/':
            nn = nn - 1

        token = tokens[:nn - 1].strip()  # 实际的token值
        tagPre = tokens[nn:].strip()  # 该token所对应的标记
        tagPre = self.TAGPRE.get(tagPre, TagPrefix.general.value) # 把原始语料的标记进行转换训练所需的标记
        if token not in self.spiltChar:
            sentence.addToken(token, tagPre)
        if token in self.spiltChar or end:
            if sentence.chars > self.max_len:  # 大于最大长度则直接过滤掉
                self.longLine += 1
            else:
                x = []
                y = []
                self.totalChars += sentence.chars
                sentence.generate_tr_line(x, y)

                if len(x) > 0 and len(x) == len(y):
                    self.datas.append(x)
                    self.labels.append(y)
                else:
                    print('处理一行数据异常, 异常行如下')
                    print(sentence.tokens)
            self.totalLine += 1
            sentence.clear()

    # 制作word2id  id2word 等
    def handlerRawData(self):
        self.df_data = pd.DataFrame({'words': self.datas, 'tags': self.labels}, index=range(len(self.datas)))
        # 　句子长度
        self.df_data['sentence_len'] = self.df_data['words'].apply(
            lambda words: len(words))  # 计算每个单词的长度，放到sentence_len列中

        # 1.用 chain(*lists) 函数把多个list拼接起来
        all_words = list(chain(*self.df_data['words'].values))
        # 2.统计所有 word
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()  # 按照每个字

        set_words = sr_allwords.index
        set_ids = range(1, len(set_words) + 1)  # 注意从1开始，因为我们准备把0作为填充值
        # tags = ['x', 's', 'b', 'm', 'e']
        tags = ['x']  # padding的时候对应的标签

        for _, memberPre in TagPrefix.__members__.items():
            for _, memberSuf in TagSurfix.__members__.items():
                if memberSuf is TagSurfix.S and memberPre is TagPrefix.general:
                    tags.append(memberPre.value + memberSuf.value)
                elif memberSuf != TagSurfix.S:
                    tags.append(memberPre.value + memberSuf.value)

        print(tags)

        tag_ids = range(len(tags))

        # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
        self.word2id = pd.Series(set_ids, index=set_words)
        self.id2word = pd.Series(set_words, index=set_ids)
        self.id2word[len(set_ids) + 1] = '<NEW>'  # 添加一个未知字符的支持
        self.word2id['<NEW>'] = len(set_ids) + 1

        self.tag2id = pd.Series(tag_ids, index=tags)
        self.id2tag = pd.Series(tags, index=tag_ids)

        self.df_data['X'] = self.df_data['words'].apply(self.X_padding)
        self.df_data['y'] = self.df_data['tags'].apply(self.y_padding)

        # 最后得到了所有的数据
        self.X = np.asarray(list(self.df_data['X'].values))
        self.y = np.asarray(list(self.df_data['y'].values))
        print('X.shape={}, y.shape={}'.format(self.X.shape, self.y.shape))
        print('Example of words: ', self.df_data['words'].values[0])
        print('Example of X: ', self.X[0])
        print('Example of tags: ', self.df_data['tags'].values[0])
        print('Example of y: ', self.y[0])

        # 保存数据
        with open(self.save_path, 'wb') as outp:
            pickle.dump(self.X, outp)
            pickle.dump(self.y, outp)
            pickle.dump(self.word2id, outp)
            pickle.dump(self.id2word, outp)
            pickle.dump(self.tag2id, outp)
            pickle.dump(self.id2tag, outp)
        print('** Finished saving the data.')

    def X_padding(self, words):
        """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
        ids = list(self.word2id[words])
        if len(ids) >= self.max_len:  # 长则弃掉
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))  # 短则补全
        return ids

    def y_padding(self, tags):
        """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
        ids = list(self.tag2id[tags])
        if len(ids) >= self.max_len:  # 长则弃掉
            return ids[:self.max_len]
        ids.extend([0] * (self.max_len - len(ids)))  # 短则补全
        return ids

    # 划分训练集、测试集、验证集
    def builderTrainData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print(
            'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
                X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

        print('Creating the data generator ...')
        data_train = BatchGenerator(X_train, y_train, shuffle=True)
        data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
        data_test = BatchGenerator(X_test, y_test, shuffle=False)
        print('Finished creating the data generator.')

        return data_train, data_valid, data_test


# 构造一个生成batch数据的类
class BatchGenerator(object):

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


if __name__ == '__main__':
    data = DataHandler(rootDir='corpus\\2014', save_path='data\\data.pkl')
    data.loadData()

    data.builderTrainData()
    print(data.X)
    print(type(data.X))
    print(data.X.shape) 