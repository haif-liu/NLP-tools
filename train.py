#encoding=utf8
import time
import tensorflow as tf
from tensorflow.contrib import crf
import data_read as readData
import bi_lstm_model as modelDef
import numpy as np

# 增加一个参数，--使分别
# 训练seg、NER、parse(seq2seq)

class BiLSTMTrain(object):
    def __init__(self, data_train=None, data_valid=None, data_test=None,
                 model=None):
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.model = model

    def train(self):
        # 设置 GPU 按需增长
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # 进行实际的训练
        sess.run(tf.global_variables_initializer())
        decay = 0.85
        max_epoch = 5
        tr_batch_size = 128
        max_max_epoch = 6  # 最大训练的epoch量
        display_num = 5  # 每个 epoch 显示是个结果
        tr_batch_num = int(self.data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
        display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
        saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
        for epoch in range(max_max_epoch):  # 整个数据集循环批次
            _lr = 1e-4
            if epoch > max_epoch:
                _lr = _lr * ((decay) ** (epoch - max_epoch))
            print('EPOCH %d， lr=%g' % (epoch + 1, _lr))
            start_time = time.time()
            _losstotal = 0.0
            show_loss = 0.0
            for batch in range(tr_batch_num):  # 一个大批次训练下面的小批次
                fetches = [self.model.loss, self.model.train_op]
                X_batch, y_batch = self.data_train.next_batch(tr_batch_size)

                feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: _lr,
                             self.model.batch_size: tr_batch_size,
                             self.model.keep_prob: 0.5}
                _loss, _ = sess.run(fetches, feed_dict)  # 每批次平均损失
                _losstotal += _loss
                show_loss += _loss
                if (batch + 1) % display_batch == 0:
                    valid_acc = self.test_epoch(self.data_valid, sess)  # valid
                    print('\ttraining loss=%g ;  valid acc= %g ' % (show_loss / display_batch,
                                                                             valid_acc))
                    show_loss = 0.0
            mean_loss = _losstotal / tr_batch_num
            if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
                save_path = saver.save(sess, self.model.model_save_path, global_step=(epoch + 1))
                print('the save path is ', save_path)
                print('词向量为：')
                print(sess.run(self.model.embedding))
            print('\ttraining %d, loss=%g ' % (self.data_train.y.shape[0], mean_loss))
            print('Epoch training %d, loss=%g, speed=%g s/epoch' % (
                self.data_train.y.shape[0], mean_loss, time.time() - start_time))

        # testing
        print('**TEST RESULT:')
        test_acc = self.test_epoch(self.data_test, sess)
        print('**Test %d, acc=%g' % (self.data_test.y.shape[0], test_acc))
        sess.close()

    def test_epoch(self, dataset=None, sess=None):
        
        _batch_size = 500
        _y = dataset.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)  # 循环批次
        correct_labels = 0
        total_labels = 0
        fetches = [self.model.scores, self.model.length, self.model.transition_params]

        for i in range(batch_num):
            X_batch, y_batch = dataset.next_batch(_batch_size)
            feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: 1e-5,
                         self.model.batch_size: _batch_size,
                         self.model.keep_prob: 1.0}

            test_score, test_length, transition_params = sess.run(fetches=fetches,
                                                                  feed_dict=feed_dict)
            for tf_unary_scores_, y_, sequence_length_ in zip(
                    test_score, y_batch, test_length):
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]
                viterbi_sequence, _ = crf.viterbi_decode(
                    tf_unary_scores_, transition_params)
                # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_


        accuracy = correct_labels / float(total_labels)
        return accuracy


if __name__ == '__main__':
    data = readData.DataHandler(rootDir='\\corpus\\2014')
    print('语料加载完成！')
    data.loadData()
    data_train, data_valid, data_test = data.builderTrainData()  # 拆分开训练集、验证集、测试集
    print('训练集、验证集、测试集拆分完成！')

    model = modelDef.BiLSTMModel(max_len=data.max_len, vocab_size=data.word2id.__len__(), class_num= data.tag2id.__len__())
    print('模型定义完成！')

    train = BiLSTMTrain(data_train, data_valid, data_test, model)
    train.train()
    print('模型训练完成！')
