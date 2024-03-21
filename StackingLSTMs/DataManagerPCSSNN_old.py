#-*-coding:utf-8-*-
import time
import random
import glob
import math
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from pandas import read_csv

class DataManager(object):
    def __init__(self, dataset, grained=3):
        self.dataTrain = []
        self.dataTest = []
        self.dataset = dataset

    #train为 [{'seqs':[[pos],...,[pos],[pos]], 'solution':[travle time]} ,...,   {},{}]
    #pos  route data
    #数据源包含所有link
    def gen_data1(self, specialLinks, scaler0, scaler1, num):
        #self.__oneLinktofile(specialLinks)
        print(0)
        allink = specialLinks[0]
        print(0)
        dataTrain, dataTest = self.__gen_one_lstm_data(allink, scaler0, scaler1, num)
        for i in range(len(dataTrain)):
            item= {}
            item['seqs_'+allink] = dataTrain[i]['seqs']
            item['solution'] = dataTrain[i]['solution']
            self.dataTrain.append(item)

        for i in range(len(dataTest)):
            item= {}
            item['seqs_'+allink] = dataTest[i]['seqs']
            item['solution'] = dataTest[i]['solution']
            self.dataTest.append(item)

        for allink in specialLinks[1:]:
            dataTrain, dataTest = self.__gen_one_lstm_data(allink, MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1)), num)
            for i in range(len(dataTrain)):
                self.dataTrain[i]['seqs_'+allink] = dataTrain[i]['seqs']
            for i in range(len(dataTest)):
                self.dataTest[i]['seqs_'+allink] = dataTest[i]['seqs']
        return self.dataTrain, self.dataTest

    def __gen_one_lstm_data(self, allink, scaler0, scaler1, num):

        # LinkRef,LinkDescription,Date,TimePeriod,AverageJT,AverageSpeed,DataQuality,LinkLength,Flow
        # AL100,A414 between M1 J7 and A405 (AL100),2015-03-01 00:00:00,8,164.20,98.88,3,4.5100002290000001,11.00
        # num = 5  #训练序列加上观察值，共num个元素为一个样本
        train_filename = self.dataset +'/' + allink+'.csv'
        dataframe = read_csv(train_filename, header=0, usecols=[3,4])
        dataset = dataframe.values.astype('float32')

        averageJT = dataset[:, 1]

        #plt.plot( MinMaxScaler(feature_range=(0, 1)).fit_transform(averageJT[0:100]))

        timePeriod = dataset[:, 0]
        averageJT = averageJT.reshape(len(averageJT), -1)
        timePeriod = timePeriod.reshape(len(timePeriod), -1)

        averageJT = scaler0.fit_transform(averageJT)

        timePeriod = scaler1.fit_transform(timePeriod)

        # 第0个数据
        # num为终点
        # split into train and test sets
        train_size = int(len(averageJT) * 0.90)
        dataTrain = []
        dataTest = []

        for i in range(num, train_size):
            # 一个训练样本的初始化
            item = {}
            item['seqs'] = averageJT[i - num:i].reshape(num,1)
            item['solution'] = averageJT[i]
            item['tar_time'] = timePeriod[i]
            dataTrain.append(item)

        for i in range(train_size, len(averageJT)):
            if averageJT[i]!=0.0:
                item = {}
                item['seqs'] = averageJT[i - num:i].reshape(num,1)
                item['solution'] = averageJT[i]
                item['tar_time'] = timePeriod[i]
                dataTest.append(item)
        return dataTrain, dataTest

    def gen_stat_link(self):
        train_filename = self.dataset+'/newMAR15.csv'
#LinkRef,LinkDescription,Date,TimePeriod,AverageJT,AverageSpeed,DataQuality,LinkLength,Flow
        #AL100,A414 between M1 J7 and A405 (AL100),2015-03-01 00:00:00,8,164.20,98.88,3,4.5100002290000001,11.00
        #num = 5  #训练序列加上观察值，共num个元素为一个样本

        with open(train_filename) as f:
            sentences = f.readlines()
            linkstat = {}
            nlink = 0
            jt = 0.0
            for i in range(1, len(sentences)):

                line = sentences[i].strip().split(',')
                linepre = sentences[i-1].strip().split(',')

                #判断是否能构成一个训练样本，如果能则生成一个训练样本

                if(line[0] == linepre[0] ):
                    nlink = nlink+1
                    jt = jt + float(line[4])
                    linkstat[line[0]] = (jt/nlink)/float(line[7])
                    #linkstat["1"+line[0]] = str(jt/nlink)+":"+line[7]
                else:
                    jt = float(line[4])
                    nlink = 1

        links = sorted(linkstat.items(),key=lambda item:item[1])
        return links[0], links[len(links)/4*3], links[-1]
        #return linkstat

    def __oneLinktofile(self, specialLinks):
        train_filename = self.dataset + '/MAR15.csv'

        for allink in specialLinks:
            # LinkRef,LinkDescription,Date,TimePeriod,AverageJT,AverageSpeed,DataQuality,LinkLength,Flow
            # AL100,A414 between M1 J7 and A405 (AL100),2015-03-01 00:00:00,8,164.20,98.88,3,4.5100002290000001,11.00
            # num = 5  #训练序列加上观察值，共num个元素为一个样本
            newsen =[];
            with open(train_filename) as f:
                sentences = f.readlines()
                newsen.append(sentences[0])
                for i in range(0, len(sentences)):

                    line = sentences[i].strip().split(',')


                    if (line[0] != 'LinkRef' and line[0] == allink):
                        newsen.append(sentences[i])
                f.close()
            with open('../datasets1/'+allink+'.csv', 'w') as f:
                f.writelines(newsen)
                f.close()

    def analysis(self, r_num, observ_t ,prediction_t, details, strtestortrain):
        # Write to the file
        with open('%s.txt' % ("lstm"), 'a') as f:
            l1 = details["test_rmse"]
            l2 = details["test_mae"]
            l3 = details["test_mape"]
            pdata = str(r_num) + ":"
            pdata += str(min(l1)) + "\t" + str(l1.index(min(l1))) + "\t"
            pdata += str(min(l2)) + "\t" + str(l2.index(min(l2))) + "\t"
            pdata += str(min(l3)) + "\t" + str(l3.index(min(l3))) + "\t"
            # for i, val in enumerate(details["test_mape"]):
            #    pdata += '(' + str(i) + "," + str(val) + ")"
            f.writelines(pdata)
            f.writelines('\n')
            '''
            pdata = str(r_num) + ":"
            for key, val in enumerate(l3):
                pdata += "(" + str(key+1) + "," + str(val) + ")"
            f.writelines(pdata)
            f.writelines('\n')
            '''
            f.close()

        # calculate root mean squared error
        rmse = math.sqrt(mean_squared_error(observ_t, prediction_t))
        mae = math.sqrt(mean_absolute_error(observ_t, prediction_t))
        mapelist = abs(observ_t-prediction_t) / observ_t
        mape = mapelist.sum() / len(mapelist)

        print('Test Score: %.4f RMSE' % (rmse))
        print('Test Score: %.4f MAE' % (mae))
        print('Test Score: %.4f MAPE' % (mape))

        with open('finalresult.txt', 'a') as f:
            # f.writelines(json.dumps(details))
            f.writelines(
                str(r_num)+":RMSE, MAE, MAPE," + str(rmse) + "," + str(mae) + "," + str(mape))
            f.writelines('\n')

            pdata = ""
            for i, val in enumerate(observ_t):
                pdata += '(' + str(i) + "," + str(val) + ")"
            f.writelines(pdata)
            pdata = ""
            for i, val in enumerate(prediction_t):
                pdata += '(' + str(i) + "," + str(val) + ")"
            f.writelines('\n')
            f.writelines(pdata)
            f.writelines('\n')

            f.close()

        # f1 = file('shoplist.data', 'w')
        # p.dump(model.params, f1)  # dump the object to a file
        # f1.close()

#        plt.plot(observ_t)
#        plt.plot(prediction_t)
#        plt.show()

    def gen_dataforRL(self):
        train_filename = self.dataset + '/newMAR15.csv'
        # LinkRef,LinkDescription,Date,TimePeriod,AverageJT,AverageSpeed,DataQuality,LinkLength,Flow
        # AL100,A414 between M1 J7 and A405 (AL100),2015-03-01 00:00:00,8,164.20,98.88,3,4.5100002290000001,11.00
        # num = 5  #训练序列加上观察值，共num个元素为一个样本
        train_data = []
        test_data = []
        with open(train_filename) as f:
            sentences = f.readlines()
            f.close()
            sentences = [i for i in sentences if i.find("LinkRef") == -1]

            # 训练数据
            utrain = {}
            utt0 = []
            for i in range(0, 96):
                key = str(i)
                utrain[key] = 0.0
                utrain[key + "L"] = 0.0

            xtrain = [[int(i.split(',')[3]), float(i.split(',')[4])] for i in sentences
                      if i.find('2015-03-31') == -1]
            ytrain = [i for i in xtrain]
            # 求utrain
            for i in ytrain:
                key = str(i[0])

                if utrain[key + 'L'] == 0:
                    utt0.append(float(i[1]))
                else:
                    utt0.append(utrain[key] / utrain[key + 'L'])
                utrain[key] = utrain[key] + float(i[1])
                utrain[key + 'L'] = utrain[key + 'L'] + 1

            xtrain.__delitem__(-1)
            ytrain.__delitem__(0)
            utt0.__delitem__(0)

            for i in range(len(xtrain)):
                train_data.append([xtrain[i][1], utt0[i], ytrain[i][0], ytrain[i][1]])

            # 测试数据
            utt1 = []
            xtest = [[int(i.split(',')[3]), float(i.split(',')[4])] for i in sentences
                     if i.find('2015-03-31') != -1]
            ytest = [i for i in xtest]
            # 求utrain
            for i in ytest:
                key = str(i[0])
                utt1.append(utrain[key] / utrain[key + 'L'])
                utrain[key] = utrain[key] + float(i[1])
                utrain[key + 'L'] = utrain[key + 'L'] + 1

            xtest.__delitem__(-1)
            ytest.__delitem__(0)
            utt1.__delitem__(0)

            for i in range(len(xtest)):
                test_data.append([xtest[i][1], utt1[i], ytest[i][0], ytest[i][1]])


        return train_data, test_data
