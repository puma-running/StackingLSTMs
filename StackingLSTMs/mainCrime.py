#-*-coding:utf-8-*-
import argparse
import math
import random
import sys
import time

import numpy as np
import theano
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

sys.path.append('/home/zhou/PycharmProjects/TravelNew')
from CrimeDataAggregation import DataManager
from LstmPCSSNN.Lstm_tree_share_lrfore import Lstm as Model
from Optimizer import OptimizerList

def train(model, train_data, optimizer, epoch_num, batch_size, train_solution):
    batch_n = (len(train_data) - 1) / batch_size + 1
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0


    for batch in xrange(batch_n):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))

        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], train_solution, optimizer)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum[0]/batch_total_nodes

#一次训练 返回 batch_loss
def do_train(model, train_data, train_solution, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0])
    total_nodes = 0
    #我认为是随机，批量的梯度下降，本次的参数的梯度赋值为0
    for _, grad in model.grad.iteritems():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                dtype=theano.config.floatX))

    for item in train_data:
        solution = [item['solution']]
        #seq1, seq2, seq3, seq4 = item['seqs_' + LSOAcodes[0]], item['seqs_' + LSOAcodes[1]], item['seqs_' + LSOAcodes[2]], item['seqs_' + LSOAcodes[3]]
        seq1, seq2, seq3, seq4 = item['seqs_1'], item['seqs_2'], item['seqs_3'], item['seqs_4']
        batch_loss_for = model.func_train(seq2, seq3, seq4, seq1, solution)
        batch_loss += np.array(batch_loss_for)
        total_nodes += 1


    # 批量的梯度求均值
    for _, grad in model.grad.iteritems():
        grad.set_value((grad.get_value() / len(train_data)).astype(theano.config.floatX))

    #根据梯度 更新params
    optimizer.iterateADAGRAD(model.grad)
    return batch_loss, total_nodes

def test(model, test_data , test_solution):
    def mae(solution,pred):
        return abs(pred - solution)

    def mape(solution, pred):
        return abs(solution - pred)/solution

    def mse(solution, pred):
        return math.pow(solution - pred, 2)

    prediction_t = []
    observ_t = []

    for item in test_data:
        solution = item['solution']
        #seq1, seq2, seq3, seq4= item['seqs_' + LSOAcodes[0]], item['seqs_' + LSOAcodes[1]], item['seqs_' + LSOAcodes[2]], item['seqs_' + LSOAcodes[3]]
        seq1, seq2, seq3, seq4 = item['seqs_1'], item['seqs_2'], item['seqs_3'], item['seqs_4']
        pred = model.func_test( seq2, seq3, seq4, seq1)

        prediction_t.append(pred)
        observ_t.append(solution)

    return observ_t, prediction_t

for i in xrange(1,11):
    if __name__ == '__main__':
        argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
        parser.add_argument('--dim_gram', type=int, default=1)
        parser.add_argument('--dataset', type=str, default='../crimedata')
        parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
        parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
        parser.add_argument('--optimizer', type=str, default='ADAGRAD')
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--lr_word_vector', type=float, default=0.5)
        parser.add_argument('--epoch', type=int, default=25)
        parser.add_argument('--batch', type=int, default=25)
        args, _ = parser.parse_known_args(argv)

        '''
        xx=[1, 2, 3, 10]
        yy=[10,20,50,100]
        mm1 = MinMaxScaler(feature_range=(0, 1))
        res = mm1.fit_transform(xx)
        res1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(yy)
        xx1 = mm1.inverse_transform(res)
        yy1 = mm1.inverse_transform(res1)
        '''
        # 输入数据处理，即向量化
        data = DataManager(args.dataset)
        #specialLinks = ['AL2202', 'AL1900', 'AL2192']  # ,'AL484A','AL3071']
        #E01032740, 2803
        #E01032739, 11888
        #E01000001, 632
        #E01000003, 162
        #E01000002, 811
        #E01000005, 1329
        #E01002704, 186

        #LSOAcodes1 = ['E01032740', 'E01032739', 'E01000001', 'E01000003']
        #LSOAcodes2 = ['E01032739', 'E01032740', 'E01000001', 'E01000003']
        #LSOAcodes3 = ['E01000001', 'E01032739', 'E01032740', 'E01000003']
        #LSOAcodes4 = ['E01000003', 'E01000001', 'E01032739', 'E01000002']
        #LSOAcodes5 = ['E01000002', 'E01000003', 'E01000001', 'E01000005']
        #LSOAcodes6 = ['E01000005', 'E01000002', 'E01000003', 'E01002704']
        #LSOAcodes7 = ['E01002704', 'E01000005', 'E01000002', 'E01000003']



        #specialLinks = ['AL292', 'AL282','AL2274','AL286']
        #specialLinks = ['AL284', 'AL281', 'AL279']
        # 15-min, 起点:1,...,r_num为输入序列，终点:r_num+1 为预测值。譬如4，1,2,3,4为输入，5为预测值
        # 譬如30-min, 3: 1;2,3;4,5,6;  r_num*2+1, r_num*2+2为预测值
        # 譬如45-min, 3: 1,2,3; 4,5,6; 7,8,9; r_num*3+1, r_num*3+2,r_num*3+3为预测值

        r_num = 7  # r_num为输入序列x的长度，不包含观察址y
        scalerJT = MinMaxScaler(feature_range=(0, 1))  # AverageJT
        scaler1 = MinMaxScaler(feature_range=(0, 1))  # TimePeriod

        #data.gen_dataFile(LSOAcodes, scalerJT, scaler1, r_num)
        LSOAcodes = ['E01032739', 'E01000001', 'E01032740', 'E01000002']
        data.gen_dataFile()
        data.gen_trainandtest(LSOAcodes, scalerJT, scaler1, r_num)
        train_data = data.dataTrain
        test_data = data.dataTest

        # ('AL1439', 28.322127489687567),
        # ('AL2202', 43.69417895312028), AL1900, AL484A, AL2192, AL3071
        # ('AL1212', 121.98061693885198)

        for i in xrange(1, 2):
            random.seed(args.seed)
            model = Model(argv)
            optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)

            details = {'train_loss': [], 'test_rmse': [], 'test_mae': [], 'test_mape': []}
            observ_t = []
            prediction_t = []

            bestepoch = 25
            # 求平均和
            train_solution = 0.0
            for item in train_data:
                train_solution = + item['solution']
            train_solution = train_solution / len(train_data)

            # 求平均和
            test_solution = 0.0
            for item in test_data:
                test_solution = + item['solution']
            test_solution = test_solution / len(test_data)

            for e in range(args.epoch):
                print e

                # for allink in specialLinks:
                random.shuffle(train_data)
                # print e
                trainloss = train(model, train_data, optimizer, e, args.batch, train_solution)
                details["train_loss"].append(trainloss)

                observ_t, prediction_t = test(model, test_data, test_solution)
                observ_t = np.array(observ_t).reshape(-1,1)

                observ_t = scalerJT.inverse_transform(observ_t)
                prediction_t = scalerJT.inverse_transform(np.array(prediction_t).reshape(len(prediction_t), -1))

                observ_t = np.array(observ_t).reshape(len(observ_t))
                prediction_t = np.array(prediction_t).reshape(len(prediction_t))

                rmse = math.sqrt(mean_squared_error(observ_t, prediction_t))
                mae = math.sqrt(mean_absolute_error(observ_t, prediction_t))
                mapelist = abs(observ_t - prediction_t) / observ_t
                mape = mapelist.sum() / len(mapelist)

                details["test_rmse"].append(rmse)
                details["test_mae"].append(mae)
                details["test_mape"].append(mape)

            data.analysis(r_num, observ_t, prediction_t, details)