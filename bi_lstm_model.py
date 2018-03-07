#encoding=utf8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

'''
# 增加一个参数，--训练seg、NER、parse(seq2seq)
#
# 模型层：输入层-->嵌入层-->双端长短记忆网络-->输出层
'''


class BiLSTMModel(object):
    def __init__(self, max_len=200, vocab_size=5159, class_num=5, model_save_path='./ckpt/bi-lstm.ckpt'):
        # config
        self.timestep_size = self.max_len = max_len  # 单个输入的句子长度;也即一次完整的样本输入的大小
        self.vocab_size = vocab_size  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = self.embedding_size = 64  # 字向量长度
        self.class_num = class_num  # 标记类型
        self.hidden_size = 128  # 隐含层节点数
        self.layer_num = 2  # bi-lstm 层数
        self.max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.model_save_path = model_save_path  # 模型保存位置
        # 获取字向量 (通过模型自我学习嵌入矩阵)
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [vocab_size, self.embedding_size], dtype=tf.float32)
        self.trainModel()

    # 权重项
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 偏置项
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # lstm单元
    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    # 双端lstm
    def bi_lstm(self, X_inputs):
        # 实际传入的参数以及转换后的输出如下:
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)

        # 输入的句子依然是padding补齐的数据
        # 计算每个句子的实际长度， 也即非0非padding部分的实际长度
        self.length = tf.reduce_sum(tf.sign(X_inputs), 1)
        self.length = tf.cast(self.length, tf.int32)

        # cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        # cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)

        # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
        #                                             sequence_length=self.length, dtype=tf.float32)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell(), self.lstm_cell(), 
                                                       inputs, sequence_length=self.length, dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, self.hidden_size * 2])
        return output  # [-1, hidden_size*2]


    # 设置模型的训练
    def trainModel(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')  # 输入
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')  # 对应的标记

        bilstm_output = self.bi_lstm(self.X_inputs) # 返回的是隐藏层状态的输出

        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b  # 隐藏层状态加上softmax才是实际的y预测值
            print(self.y_pred.shape)
            self.scores = tf.reshape(self.y_pred, [-1, self.timestep_size,
                                              self.class_num])  # 算好分数后，再重新reshape成[batchsize, timesteps, num_class]
            print(self.scores.shape)
            log_likelihood, self.transition_params = crf.crf_log_likelihood(self.scores, self.y_inputs, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)


        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        print('Finished training the bi-lstm model.')