#-*-coding:utf-8-*-
import math
import random
import time

import numpy as np
from NaiveLSTM import NaiveLSTM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# sys.path.append('D:\pythoncode\Paper\TravelTime0307')
from DataManagerPCSSNN_old import DataManager
# from Standard_Lstm_Inputs import Lstm as Model
from GridLSTM import GridLSTM as GridLSTM
import torch.optim as optim
# from Optimizer import OptimizerList
import torch
from args import args
from GridHook import My_hook_lstm, My_hook_linear, Hook_process

def train(model, train_data, batch_size):
    batch_n = int((len(train_data) - 1) / batch_size + 1)
    st_time = time.time()
    loss_sum = torch.zeros(1).to(args.device)
    total_nodes = torch.zeros(1).to(args.device)

    args.optimizer = optim.Adam(model.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay)
    # 查看模型参数
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    for batch in range(batch_n):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))

        batch_loss, batch_total_nodes = do_train(model, train_data[start:end])
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum/batch_total_nodes

#一次训练 返回 batch_loss
def do_train(model, train_data):
    # eps0 = 1e-8
    #我认为是随机，批量的梯度下降，本次的参数的梯度赋值为0
    # for _, grad in model.grad.iteritems():
    #     grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
    #             dtype=theano.config.floatX))
    # 0. batch_loss, optimizer
    total_nodes = torch.tensor(0).to(args.device)
    batch_loss = torch.tensor(0.0).to(args.device)
    args.optimizer.zero_grad()
    # 1. forward
    for item in train_data:
        solution = torch.tensor(np.array([item['solution']])).to(args.device)
        # specialLinks = ['AL292', 'AL282','AL2274','AL286']
        #seq292, seq282, seq2274, seq286 =  item['seqs_'+specialLinks[0]], item['seqs_'+specialLinks[1]], item['seqs_'+specialLinks[2]], item['seqs_'+specialLinks[3]]
        #batch_loss_for = model.func_train(seq286, seq282, seq2274, seq292, solution)

        #specialLinks = ['AL282', 'AL283', 'AL2274', 'AL292']
        seq1, seq2, seq3, seq4 = item['seqs_' + specialLinks[0]], item['seqs_' + specialLinks[1]], item['seqs_' + specialLinks[2]], item['seqs_' + specialLinks[3]]
        # seqs = np.concatenate((seq1,seq2,seq3,seq4),1)
        seq1 = np.expand_dims(seq1, axis=0)
        seq1 = torch.tensor(seq1).to(args.device)
        h_list1, c_list1 = model(seq1)

        # h_list = model.linear2(h_list)
        # h_list = h_list.unsqueeze(0)
        seq2 = np.expand_dims(seq2, axis=0)
        seq2 = torch.tensor(seq2).to(args.device)
        h_list2, c_list2 = model(seq2, hc2_list=(h_list1, c_list1))
        # h_list = model.linear2(h_list)
        # h_list = h_list.unsqueeze(0)
        seq3 = np.expand_dims(seq3, axis=0)
        seq3 = torch.tensor(seq3).to(args.device)
        h_list3, c_list3 = model(seq3, hc2_list=(h_list2, c_list2))
        # h_list = h_list.unsqueeze(0)
        seq4 = np.expand_dims(seq4, axis=0)
        seq4 = torch.tensor(seq4).to(args.device)
        h_list4, c_list4 = model(seq4, hc2_list=(h_list3, c_list3))

        arrH = torch.cat((h_list1.squeeze(0),h_list2.squeeze(0),h_list3.squeeze(0),h_list4.squeeze(0)),dim=1)
        pred_for_train = model.linear(arrH.view(-1))
        pred_for_train = pred_for_train.unsqueeze(0)
        # solution
        batch_loss_for = args.criterion(solution, pred_for_train)

        #batch_loss_for = model.func_train(seq1, solution)
        #batch_loss_for = model.func_train(np.asarray(seq1,dtype = np.float32), solution)
        batch_loss += batch_loss_for
        total_nodes += 1
    # 2. Backward
    batch_loss.backward()
    # 3. Update
    args.optimizer.step()

    # # 批量的梯度求均值
    # for _, grad in model.grad.iteritems():
    #     grad.set_value((grad.get_value() / len(train_data)).astype(theano.config.floatX))

    # #根据梯度 更新params
    # optimizer.iterateADAGRAD(model.grad)
    return batch_loss, total_nodes

def test(model, test_data):
    # def mae(solution,pred):
    #     return abs(pred - solution)
    # def mape(solution, pred):
    #     return abs(solution - pred)/solution
    # def mse(solution, pred):
    #     return math.pow(solution - pred, 2)
    prediction_t = torch.empty(0).to(args.device)
    observ_t = torch.empty(0).to(args.device)
    model.eval()
    with torch.no_grad():
        for item in test_data:
            solution = torch.tensor(np.array([item['solution']])).to(args.device)
            #specialLinks = ['AL292', 'AL282','AL2274','AL286']
            #seq292, seq282, seq2274, seq286 =  item['seqs_'+specialLinks[0]], item['seqs_'+specialLinks[1]], item['seqs_'+specialLinks[2]], item['seqs_'+specialLinks[3]]
            #pred = model.func_test(seq286, seq282, seq2274, seq292)

            #specialLinks = ['AL282', 'AL283', 'AL2274', 'AL292']
            seq1, seq2, seq3, seq4 = item['seqs_' + specialLinks[0]], item['seqs_' + specialLinks[1]], item['seqs_' + specialLinks[2]], item['seqs_' + specialLinks[3]]
            # seqs = np.concatenate((seq1,seq2,seq3,seq4),1)
            seq1 = np.expand_dims(seq1, axis=0)
            seq1 = torch.tensor(seq1).to(args.device)
            h_list1, c_list1 = model(seq1)

            # h_list = model.linear2(h_list)
            # h_list = h_list.unsqueeze(0)
            seq2 = np.expand_dims(seq2, axis=0)
            seq2 = torch.tensor(seq2).to(args.device)
            h_list2, c_list2 = model(seq2, hc2_list=(h_list1, c_list1))
            # h_list = model.linear2(h_list)
            # h_list = h_list.unsqueeze(0)
            seq3 = np.expand_dims(seq3, axis=0)
            seq3 = torch.tensor(seq3).to(args.device)
            h_list3, c_list3 = model(seq3, hc2_list=(h_list2, c_list2))
            # h_list = h_list.unsqueeze(0)
            seq4 = np.expand_dims(seq4, axis=0)
            seq4 = torch.tensor(seq4).to(args.device)
            h_list4, c_list4 = model(seq4, hc2_list=(h_list3, c_list3))

            arrH = torch.cat((h_list1.squeeze(0),h_list2.squeeze(0),h_list3.squeeze(0),h_list4.squeeze(0)),dim=1)
            pred_for_test = model.linear(arrH.view(-1))
            #pred = model.func_test(seq1)
            # if len(prediction_t)==0:
            prediction_t = torch.cat([prediction_t, pred_for_test], dim=0)
            observ_t = torch.cat([observ_t, solution], dim=0)
    return observ_t, prediction_t
for i in range(1,2):
    if __name__ == '__main__':
        random.seed(args.seed)
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
        specialLinks = ['AL282', 'AL2274','AL292','AL286']
        #specialLinks = ['AL3070', 'AL3069A', 'AL2202', 'AL1900', 'AL1896', 'AL1891', 'AL1885','AL1883','AL1877','AL2991']

        r_num = 7  # r_num为输入序列x的长度，不包含观察址y
        scalerJT = MinMaxScaler(feature_range=(0, 1))  # AverageJT
        scaler1 = MinMaxScaler(feature_range=(0, 1))  # TimePeriod

        train_data, test_data = data.gen_data1(specialLinks, scalerJT, scaler1, r_num)

        # ('AL1439', 28.322127489687567),
        # ('AL2202', 43.69417895312028), AL1900, AL484A, AL2192, AL3071
        # ('AL1212', 121.98061693885198)

        model = GridLSTM().to(args.device)
        my_hook_lstm = My_hook_lstm()
        model.register_forward_hook(my_hook_lstm.forward_hook)
        my_hook_linear = My_hook_linear()
        model.linear.register_forward_hook(my_hook_linear.forward_hook) 
        # optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)

        details = {'train_loss': [], 'test_rmse': [], 'test_mae': [], 'test_mape': []}
        observ_t = []
        prediction_t = []
        bestepoch = 25
        # # 求平均和
        # train_solution = 0.0
        # for item in train_data:
        #     train_solution = + item['solution']
        # train_solution = train_solution / len(train_data)

        # # 求平均和
        # test_solution = 0.0
        # for item in test_data:
        #     test_solution = + item['solution']
        # test_solution = test_solution / len(test_data)

        for e in range(args.epoch):
            print(e)
            # for allink in specialLinks:
            random.shuffle(train_data)
            # print e
            trainloss = train(model, train_data, args.batch)
            details["train_loss"].append(trainloss)

            observ_t, prediction_t = test(model, test_data)
            observ_t = np.array(observ_t.cpu()).reshape(-1,1)
            prediction_t = np.array(prediction_t.cpu()).reshape(-1,1)

            observ_t = scalerJT.inverse_transform(observ_t)
            prediction_t = scalerJT.inverse_transform(prediction_t)
            observ_t = observ_t.reshape(-1)
            prediction_t = prediction_t.reshape(-1)
            # observ_t = np.array(observ_t).reshape(len(observ_t))
            # prediction_t = np.array(prediction_t).reshape(len(prediction_t))

            rmse = math.sqrt(mean_squared_error(observ_t, prediction_t))
            mae = math.sqrt(mean_absolute_error(observ_t, prediction_t))
            mapelist = abs(observ_t - prediction_t) / observ_t
            mape = mapelist.sum() / len(mapelist)

            details["test_rmse"].append(rmse)
            details["test_mae"].append(mae)
            details["test_mape"].append(mape)
        
        _hook_process = Hook_process()
        _hook_process.process_hook(my_hook_lstm, my_hook_linear)
        data.analysis(r_num, observ_t, prediction_t, details, "test")

