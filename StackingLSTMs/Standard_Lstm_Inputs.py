# -*-coding:utf-8-*-
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import time
import collections
import math


class Lstm(object):
    def __init__(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000 * time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=1)
        parser.add_argument('--dim_hidden', type=int, default=4)

        args, _ = parser.parse_known_args(argv)

        self.name = args.name
        self.srng = RandomStreams(seed=args.rseed)
        self.dim_word, self.dim_hidden = args.dim_word, args.dim_hidden

        self.init_param()
        self.init_function()

    def init_param(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        u = lambda x: 1 / np.sqrt(x)

        dimc, dimh = self.dim_word, self.dim_hidden

        u_dim = 1

        dim_lstm_para2 = dimh + dimc
        self.Wi2 = shared_matrix((dimh, dim_lstm_para2), 'Wi2', u(u_dim))
        self.Wo2 = shared_matrix((dimh, dim_lstm_para2), 'Wo2', u(u_dim))
        self.Wf2 = shared_matrix((dimh, dim_lstm_para2), 'Wf2', u(u_dim))
        self.Wc2 = shared_matrix((dimh, dim_lstm_para2), 'Wc2', u(u_dim))
        self.bi2 = shared_matrix((dimh,), 'bi2', 0.)  # 是不是列向量？
        self.bo2 = shared_matrix((dimh,), 'bo2', 0.)
        self.bf2 = shared_matrix((dimh,), 'bf2', 0.)
        self.bc2 = shared_matrix((dimh,), 'bc2', 0.)

        dim_lstm_para1 = dimh+dimh
        self.Wi1 = shared_matrix((dimh, dim_lstm_para1), 'Wi1', u(u_dim))
        self.Wo1 = shared_matrix((dimh, dim_lstm_para1), 'Wo1', u(u_dim))
        self.Wf1 = shared_matrix((dimh, dim_lstm_para1), 'Wf1', u(u_dim))
        self.Wc1 = shared_matrix((dimh, dim_lstm_para1), 'Wc1', u(u_dim))
        self.bi1 = shared_matrix((dimh,), 'bi1', 0.)  # 是不是列向量？
        self.bo1 = shared_matrix((dimh,), 'bo1', 0.)
        self.bf1 = shared_matrix((dimh,), 'bf1', 0.)
        self.bc1 = shared_matrix((dimh,), 'bc1', 0.)

        #self.Ws1 = shared_matrix((dimh, 1), 'Ws1', u(u_dim))
        #self.bs1 = shared_matrix((1, ), 'bs1', 0.)

        self.Ws = shared_matrix((dimh,), 'Ws', u(u_dim))
        self.bs = shared_matrix(( ), 'bs', 0.)

        self.h1, self.c1 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.h2, self.c2 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.h3, self.c3 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.h4, self.c4 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.h5, self.c5 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)

        self.params = [self.Wi2, self.Wo2, self.Wf2, self.Wc2, self.bi2, self.bo2, self.bf2, self.bc2,
                       self.Wi1, self.Wo1, self.Wf1, self.Wc1, self.bi1, self.bo1, self.bf1, self.bc1,
                       self.Ws, self.bs]

    def init_function(self):
        self.seq1_matrix = T.matrix()
        self.seq2_matrix = T.matrix()
        self.seq3_matrix = T.matrix()
        self.seq4_matrix = T.matrix()
        self.solution = T.vector()
        h1 = T.vector()
        c1 = T.vector()
        h2 = T.vector()
        c2 = T.vector()
        h3 = T.vector()
        c3 = T.vector()
        h4 = T.vector()
        c4 = T.vector()
        h5 = T.vector()
        c5 = T.vector()


        # h_p = theano.printing.Print('h')(h)
        # h, c = T.zeros_like(self.bf1, dtype=theano.config.floatX), T.zeros_like(self.bc1, dtype=theano.config.floatX)
        # h, c = T.zeros_like(self.bf, dtype=theano.config.floatX), T.zeros_like(self.bc, dtype=theano.config.floatX)
        # 神经元的操作

        def encode(x_1, h_fore, c_fore, Wf, Wi, Wo, Wc, bf, bi, bo, bc):
            #x_1 = theano.printing.Print('x_1')(x_1)
            #h_fore = theano.printing.Print('h_fore')(h_fore)
            #v = T.concatenate([h_fore, x_1, x_2, x_3, x_4])  # 为（60)是列向量
            v = T.concatenate([h_fore, x_1])  # 为（60)是列向量
            f_t = T.nnet.sigmoid(T.dot(Wf, v) + bf)
            i_t = T.nnet.sigmoid(T.dot(Wi, v) + bi)
            o_t = T.nnet.sigmoid(T.dot(Wo, v) + bo)
            c_next = f_t * c_fore + i_t * T.tanh(T.dot(Wc, v) + bc)
            #c_next = i_t * T.tanh(T.dot(Wc, v) + bc)
            h_next = o_t * T.tanh(c_next)
            return h_next, c_next

        scan_result1, _ = theano.scan(fn=encode, sequences=[self.seq1_matrix], outputs_info=[h1, c1],
                                      non_sequences=[self.Wf2, self.Wi2, self.Wo2, self.Wc2, self.bf2, self.bi2,
                                                     self.bo2, self.bc2])
        scan_result2, _ = theano.scan(fn=encode, sequences=[self.seq2_matrix], outputs_info=[h2, c2],
                                      non_sequences=[self.Wf2, self.Wi2, self.Wo2, self.Wc2, self.bf2, self.bi2,
                                                     self.bo2, self.bc2])
        scan_result3, _ = theano.scan(fn=encode, sequences=[self.seq3_matrix], outputs_info=[h3, c3],
                                      non_sequences=[self.Wf2, self.Wi2, self.Wo2, self.Wc2, self.bf2, self.bi2,
                                                     self.bo2, self.bc2])
        scan_result4, _ = theano.scan(fn=encode, sequences=[self.seq4_matrix], outputs_info=[h4, c4],
                                      non_sequences=[self.Wf2, self.Wi2, self.Wo2, self.Wc2, self.bf2, self.bi2,
                                                     self.bo2, self.bc2])

        #value = [scan_result4[0][-1]]
        value = T.concatenate([[scan_result1[0][-1]], [scan_result2[0][-1]], [scan_result3[0][-1]], [scan_result4[0][-1]]])

        scan_result5, _ = theano.scan(fn=encode, sequences=[value], outputs_info=[h5, c5],
                                      non_sequences=[self.Wf1, self.Wi1, self.Wo1, self.Wc1, self.bf1, self.bi1,
                                                     self.bo1, self.bc1])

        #value = T.concatenate([scan_result1[0], scan_result2[0], scan_result3[0], scan_result4[0]]).flatten()  # 为（60)是列向量
        #value = theano.printing.Print('value')(value)

        #pred_for_train1 = T.dot(value, self.Ws1) + self.bs1
        #pred_for_test1 = T.dot(value, self.Ws1) + self.bs1

        #pred_for_train1 = pred_for_train1.flatten()
        #pred_for_test1 = pred_for_test1.flatten()

        #pred_for_train1 = theano.printing.Print('pred_for_train1')(pred_for_train1)

        self.pred_for_train = T.dot(scan_result5[0][-1], self.Ws) + self.bs
        self.pred_for_test = T.dot(scan_result5[0][-1], self.Ws) + self.bs

        # self.pred_for_train = scan_result4[0]
        # self.pred_for_test = scan_result4[0]

        # loss
        self.loss = T.mean(T.sqr(self.solution - self.pred_for_train))

        self.updates = collections.OrderedDict()

        grads = T.grad(self.loss, self.params)

        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), \
                                         dtype=theano.config.floatX))
            self.grad[param] = g
            # 一小批次训练的梯度累加
            self.updates[g] = g + grad

        self.func_train = theano.function(inputs=[self.seq1_matrix, self.seq2_matrix,self.seq3_matrix,self.seq4_matrix, self.solution
            , theano.In(h1, value=self.h1), theano.In(c1, value=self.c1)
            , theano.In(h2, value=self.h2), theano.In(c2, value=self.c2)
            , theano.In(h3, value=self.h3), theano.In(c3, value=self.c3)
            , theano.In(h4, value=self.h4), theano.In(c4, value=self.c4)
            , theano.In(h5, value=self.h5), theano.In(c5, value=self.c5)
                                                  ],
                                          outputs=[self.loss],
                                          updates=self.updates,
                                          on_unused_input='warn')
        # , mode='DebugMode')


        self.func_test = theano.function(
            inputs=[self.seq1_matrix, self.seq2_matrix, self.seq3_matrix, self.seq4_matrix
                , theano.In(h1, value=self.h1), theano.In(c1, value=self.c1)
                , theano.In(h2, value=self.h2), theano.In(c2, value=self.c2)
                , theano.In(h3, value=self.h3), theano.In(c3, value=self.c3)
                , theano.In(h4, value=self.h4), theano.In(c4, value=self.c4)
                , theano.In(h5, value=self.h5), theano.In(c5, value=self.c5)
                    ],
            outputs=[self.pred_for_test],
            on_unused_input='warn')
