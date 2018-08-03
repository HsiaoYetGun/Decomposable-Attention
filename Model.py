'''
Created on August 1, 2018
@author : hsiaoyetgun (yqxiao)
Reference : A Decomposable Attention Model for Natural Language Inference (EMNLP 2016)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Utils import print_shape

class Decomposable(object):
    def __init__(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value):
        # model init
        self._parameter_init(seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value)
        self._placeholder_init()

        # model operation
        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)

    # init hyper-parameters
    def _parameter_init(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value):
        """
        :param seq_length: max sentence length
        :param n_vocab: word nums in vocabulary
        :param embedding_size: embedding vector dims
        :param hidden_size: hidden dims
        :param attention_size: attention dims
        :param n_classes: nums of output label class
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param optimizer: optimizer of training
        :param l2: l2 regularization constant
        :param clip_value: if gradients value bigger than this value, clip it
        """
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # Note that attention_size is not used in this model
        self.attention_size = attention_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2 = l2
        self.clip_value = clip_value

    # placeholder declaration
    def _placeholder_init(self):
        """
        premise_mask: actual length of premise sentence
        hypothesis_mask: actual length of hypothesis sentence
        embed_matrix: with shape (n_vocab, embedding_size)
        dropout_keep_prob: dropout keep probability
        :return:
        """
        self.premise = tf.placeholder(tf.int32, [None, self.seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.premise_mask = tf.placeholder(tf.float32, [None, self.seq_length], 'premise_mask')
        self.hypothesis_mask = tf.placeholder(tf.float32, [None, self.seq_length], 'hypothesis_mask')
        self.embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.embedding_size], 'embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # build graph
    def _logits_op(self):
        alpha, beta = self._attendBlock('Attend')
        v_1, v_2 = self._compareBlock(alpha, beta, 'Compare')
        logits = self._aggregateBlock(v_1, v_2, 'Aggregate')
        return logits

    # feed forward unit
    def _feedForwardBlock(self, inputs, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor with shape (batch_size, seq_length, embedding_size)
        :param num_units: dimensions of each feed forward layer
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
        """
        with tf.variable_scope(scope, reuse = isReuse):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, num_units, tf.nn.relu, kernel_initializer = initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                resluts = tf.layers.dense(outputs, num_units, tf.nn.relu, kernel_initializer = initializer)
                return resluts

    # decomposable attend block ("3.1 Attend" in paper)
    def _attendBlock(self, scope):
        """
        :param scope: scope name

        embeded_left, embeded_right: tensor with shape (batch_size, seq_length, embedding_size)
        F_a_bar, F_b_bar: output of feed forward layer (F), tensor with shape (batch_size, seq_length, hidden_size)
        attentionSoft_a, attentionSoft_b: using Softmax at two directions, tensor with shape (batch_size, seq_length, seq_length)
        e: attention matrix with mask, tensor with shape (batch_size, seq_length, seq_length)

        :return: alpha: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
                 beta: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        """
        with tf.device('/cpu:0'):
            self.Embedding = tf.get_variable('Embedding', [self.n_vocab, self.embedding_size], tf.float32)
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
            print_shape('embeded_left', self.embeded_left)
            print_shape('embeded_right', self.embeded_right)

        with tf.variable_scope(scope):
            F_a_bar  = self._feedForwardBlock(self.embeded_left, self.hidden_size, 'F')
            F_b_bar = self._feedForwardBlock(self.embeded_right, self.hidden_size, 'F', isReuse = True)
            print_shape('F_a_bar', F_a_bar)
            print_shape('F_b_bar', F_b_bar)

            # e_i,j = F'(a_hat, b_hat) = F(a_hat).T * F(b_hat) (1)
            e_raw = tf.matmul(F_a_bar, tf.transpose(F_b_bar, [0, 2, 1]))
            # mask padding sequence
            mask = tf.multiply(tf.expand_dims(self.premise_mask, 2), tf.expand_dims(self.hypothesis_mask, 1))
            e = tf.multiply(e_raw, mask)
            print_shape('e', e)

            attentionSoft_a = tf.exp(e - tf.reduce_max(e, axis=2, keepdims=True))
            attentionSoft_b = tf.exp(e - tf.reduce_max(e, axis=1, keepdims=True))
            # mask attention weights
            attentionSoft_a = tf.multiply(attentionSoft_a, tf.expand_dims(self.hypothesis_mask, 1))
            attentionSoft_b = tf.multiply(attentionSoft_b, tf.expand_dims(self.premise_mask, 2))
            attentionSoft_a = tf.divide(attentionSoft_a, tf.reduce_sum(attentionSoft_a, axis=2, keepdims=True))
            attentionSoft_b = tf.divide(attentionSoft_b, tf.reduce_sum(attentionSoft_b, axis=1, keepdims=True))
            attentionSoft_a = tf.multiply(attentionSoft_a, mask)
            attentionSoft_b = tf.transpose(tf.multiply(attentionSoft_b, mask), [0, 2, 1])
            print_shape('att_soft_a', attentionSoft_a)
            print_shape('att_soft_b', attentionSoft_b)

            # beta = \sum_{j=1}^l_b \frac{\exp(e_{i,j})}{\sum_{k=1}^l_b \exp(e_{i,k})} * b_hat_j
            # alpha = \sum_{i=1}^l_a \frac{\exp(e_{i,j})}{\sum_{k=1}^l_a \exp(e_{k,j})} * a_hat_i (2)
            beta = tf.matmul(attentionSoft_b, self.embeded_left)
            alpha = tf.matmul(attentionSoft_a, self.embeded_right)
            print_shape('alpha', alpha)
            print_shape('beta', beta)

            return alpha, beta

    # compare block ("3.2 Compare" in paper)
    def _compareBlock(self, alpha, beta, scope):
        """
        :param alpha: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        :param beta: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        :param scope: scope name

        a_beta, b_alpha: concat of [embeded_premise, beta], [embeded_hypothesis, alpha], tensor with shape (batch_size, seq_length, 2 * embedding_size)

        :return: v_1: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
                 v_2: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        """
        with tf.variable_scope(scope):
            a_beta = tf.concat([self.embeded_left, beta], axis=2)
            b_alpha = tf.concat([self.embeded_right, alpha], axis=2)
            print_shape('a_beta', a_beta)
            print_shape('b_alpha', b_alpha)

            # v_1,i = G([a_bar_i, beta_i])
            # v_2,j = G([b_bar_j, alpha_j]) (3)
            v_1 = self._feedForwardBlock(a_beta, self.hidden_size, 'G')
            v_2 = self._feedForwardBlock(b_alpha, self.hidden_size, 'G', isReuse=True)
            print_shape('v_1', v_1)
            print_shape('v_2', v_2)
            return v_1, v_2

    # composition block ("3.3 Aggregate" in paper)
    def _aggregateBlock(self, v_1, v_2, scope):
        """
        :param v_1: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        :param v_2: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        :param scope: scope name

        v1_sum, v2_sum: sum of the compared phrases (axis = seq_length), tensor with shape (batch_size, hidden_size)
        v: concat of v1_sum, v2_sum, tensor with shape (batch_size, 2 * hidden_size)
        ff_outputs: output of feed forward layer (H), tensor with shape (batch_size, hidden_size)

        :return: y_hat: output of a linear layer, tensor with shape (batch_size, n_classes)
        """
        with tf.variable_scope(scope):
            # v1 = \sum_{i=1}^l_a v_{1,i}
            # v2 = \sum_{j=1}^l_b v_{2,j} (4)
            v1_sum = tf.reduce_sum(v_1, axis=1)
            v2_sum = tf.reduce_sum(v_2, axis=1)
            print_shape('v1_sum', v1_sum)
            print_shape('v2_sum', v2_sum)

            # y_hat = H([v1, v2]) (5)
            v = tf.concat([v1_sum, v2_sum], axis=1)
            print_shape('v', v)

            ff_outputs = self._feedForwardBlock(v, self.hidden_size, 'H')
            print_shape('ff_outputs', ff_outputs)

            y_hat = tf.layers.dense(ff_outputs, self.n_classes)
            print_shape('y_hat', y_hat)
            return y_hat

    # calculate classification loss
    def _loss_op(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # define optimizer
    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op