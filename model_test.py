#encoding=utf8
import re
import os
import numpy as np
import tensorflow as tf
import data_read as readData
import bi_lstm_model as modelDef
from sentence import TagSurfix
from tensorflow.contrib import crf


class BiLSTMTest(object):
    def __init__(self, data=None, model_path='ckpt/',
                 test_file='test/test', test_result='test/test_result'):
        self.data = data
        self.model_path = model_path
        self.test_file = test_file
        self.test_result = test_result
        self.sess = tf.Session()
        self.isload = self.loadModel(self.sess)

    #  文字转ID
    def text2ids(self, text):
        """把字片段text转为 ids."""
        words = list(text)
        ids = list(self.data.word2id[words].fillna(self.data.word2id['<NEW>']))  # 找不到的词用特殊ID替换
        # ids = list(self.data.word2id[words])
        if len(ids) >= self.data.max_len:  # 长则弃掉
            print(u'输出片段超过%d部分无法处理' % (self.data.max_len))
            return ids[:self.data.max_len]
        ids.extend([0] * (self.data.max_len - len(ids)))  # 短则补全
        ids = np.asarray(ids).reshape([-1, self.data.max_len])
        return ids

    # 对于一个句子进行切分
    def simple_cut(self, text, sess=None):
        """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
        if text:
            X_batch = self.text2ids(text)  # 这里每个 batch 是一个样本
            fetches = [self.model.scores, self.model.length, self.model.transition_params]
            feed_dict = {self.model.X_inputs: X_batch, self.model.lr: 1.0, self.model.batch_size: 1,
                         self.model.keep_prob: 1.0}
            test_score, test_length, transition_params = sess.run(fetches, feed_dict) # padding填充的部分直接丢弃
            tags, _ = crf.viterbi_decode(
                test_score[0][:test_length[0]], transition_params)

            tags = list(self.data.id2tag[tags])
            words = []
            for i in range(len(text)):  # 按照标签分割开
                if tags[i] == TagSurfix.S.value or tags[i].endswith(TagSurfix.B.value):
                    if tags[i].endswith('_' + TagSurfix.B.value):
                        words.append([text[i], tags[i][:tags[i].find('_')]]) # 带上所属的标签
                    else:
                        words.append(text[i])
                else:
                    if isinstance(words[-1], list):
                        words[-1][0] += text[i]
                    else:
                        words[-1] += text[i]

            return words
        else:
            return []

    # 切分内容原文
    def cut_word(self, sentence):
        """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
        # not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
        not_cuts = re.compile(u'[。？！.\?!]')
        result = []
        start = 0

        for seg_sign in not_cuts.finditer(sentence):
            result.extend(self.simple_cut(sentence[start:seg_sign.start()], self.sess))
            result.append(sentence[seg_sign.start():seg_sign.end()])  # 追加相关分割符
            start = seg_sign.end()
        result.extend(self.simple_cut(sentence[start:], self.sess))

        return result

    # 读取测试文件并且写入测试文件
    def testfile(self):
        isFile = os.path.isfile(self.test_file)
        if isFile:
            with open(self.test_result, "w", encoding='utf-8') as out:  # 读写文件默认都是UTF-8编码的
                with open(self.test_file, "r", encoding='utf-8') as fp:
                    for line in fp.readlines():
                        print(line.strip())
                        result = self.cut_word(line.strip())
                        rss = ''
                        for each in result:
                            if isinstance(each, list):
                                rss = rss + each[0] + ' /' + each[1] + ' '
                            else:
                                rss = rss + each + ' / '

                        print(rss)
                        out.write("%s\n" % (rss))

    # 暂时未启用,这种从模型中直接恢复图使用起来比较麻烦
    def loadModel2(self, sess=None):
        isload = False
        ckpt = tf.train.latest_checkpoint(self.model_path)
        if ckpt:
            print('模型路径：' + ckpt)
            # 直接从模型元数据文件中恢复模型所构建的图
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(sess, ckpt)
            gv = [v for v in tf.global_variables()]
            for v in gv:
                print(v.name)

            self.model = modelDef.BiLSTMModel()
            isload = True

        return isload

    # 加载还原模型
    def loadModel(self, sess=None):
        isload = False
        # 再次构建模型, 然后恢复数据
        self.model = modelDef.BiLSTMModel(vocab_size=self.data.word2id.__len__(), class_num=self.data.tag2id.__len__())
        ckpt = tf.train.latest_checkpoint(self.model_path)
        print(ckpt)
        saver = tf.train.Saver()
        if ckpt:
            saver.restore(sess, ckpt)
            isload = True
        return isload


if __name__ == '__main__':
    #### 测试分词结果 ####
    sentence = '报道称，韩国统一部先后于当日上午9时和下午4时通过板门店联络渠道拨打电话，但朝方没有应答。韩朝板门店渠道自2016年2月开城工业园区停运起切断至今。'

    data = readData.DataHandler()
    data.loadData()

    test = BiLSTMTest(data)

    if test.isload:
        # result_new = test.cut_word(sentence)
        # rss = ''
        # for each in result_new:
        #     rss = rss + each + ' / '
        # print(rss)
        test.testfile()
