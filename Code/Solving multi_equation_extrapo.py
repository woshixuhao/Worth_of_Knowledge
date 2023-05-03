from pyDOE import *
from scipy.stats import uniform
import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import time
from sklearn.feature_selection import mutual_info_regression
import math


start_time=time.time()
torch.manual_seed(525)
np.random.seed(1101)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Latin hypercube sampling
def make_data(data_num):
    lhd = lhs(2, samples=data_num)
    a= uniform(loc=0, scale=math.pi).ppf(lhd[:,0])  # sample from U[0, 5)
    b=uniform(loc=-math.pi, scale=math.pi).ppf(lhd[:,1])
    c=np.abs(np.sin(a)-np.cos(b))
    d=np.log((a-b)**2+1)
    e=(1+c*c)/2
    f=np.exp(-e)
    data=np.squeeze(np.array([a,b,c,d,e,f])).T
    return data

def make_data_out(data_num):
    lhd = lhs(2, samples=data_num)
    a= uniform(loc=math.pi, scale=math.pi).ppf(lhd[:,0])  # sample from U[0, 5)
    b=uniform(loc=math.pi, scale=math.pi).ppf(lhd[:,1])
    #b = uniform(loc=0, scale=math.pi).ppf(lhd[:, 1])
    c=np.abs(np.sin(a)-np.cos(b))
    d=np.log((a-b)**2+1)
    e=(1+c*c)/2
    f=np.exp(-e)
    data=np.squeeze(np.array([a,b,c,d,e,f])).T
    return data
class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

# data_out=make_data_out(10000)
# np.save('data_save/dataset_out_close.npy',data_out)
#data_out=np.load('data_save/dataset_out_close.npy')
data_out=np.load('data_save/dataset_out.npy')
data = np.load(f'data_save/dataset.npy')

MODEL='Measurement'
data_num=100
X=torch.from_numpy(data[:,0:2].astype(np.float32))
Y=torch.from_numpy(data[:,2:].astype(np.float32))
X_out=torch.from_numpy(data_out[:,0:2].astype(np.float32))
Y_out=torch.from_numpy(data_out[:,2:].astype(np.float32))
test_num = 5000
pretrain_num=20000

def data_ini(meta_num):
    a_meta = torch.linspace(0, 2*math.pi, meta_num)
    b_meta = torch.linspace(-math.pi, 2*math.pi, meta_num)
    row_num = 0
    data_prepare = torch.zeros(2)
    database_meta = torch.zeros([meta_num*meta_num, 2])
    for j in range(meta_num):
        for i in range(meta_num):
            data_prepare[0] = a_meta[j]
            data_prepare[1] = b_meta[i]
            database_meta[row_num] = data_prepare
            row_num += 1
    database_meta = Variable(database_meta, requires_grad=True).to(device)
    return database_meta

class Loss_func():
    def __init__(self,database_meta,Net):
        self.Net=Net
        self.database_meta=database_meta
        self.a=database_meta[:,0].reshape(-1,1)
        self.b=database_meta[:,1].reshape(-1,1)
        self.c=Net(database_meta)[:,0].reshape(-1,1)
        self.d = Net(database_meta)[:, 1].reshape(-1, 1)
        self.e = Net(database_meta)[:, 2].reshape(-1, 1)
        self.f = Net(database_meta)[:, 3].reshape(-1, 1)

    def loss_data(self,x,y):
        loss=torch.mean((self.Net(x)-y) ** 2)
        return loss

    def loss_c(self):
        loss = torch.mean((self.c-torch.abs(torch.sin(self.a)-torch.cos(self.b))) ** 2)
        return loss

    def loss_d(self):
        loss = torch.mean((self.d-torch.log((self.a-self.b)**2+1)) ** 2)
        return loss

    def loss_e(self):
        loss = torch.mean((self.e-(1+self.c*self.c)/2) ** 2)
        return loss

    def loss_f(self):
        loss = torch.mean((self.f-torch.exp(-self.e)) ** 2)
        return loss

    def loss_range(self):
        loss = torch.mean((torch.relu(-self.f)) ** 2)+torch.mean((torch.relu(-self.e)) ** 2)
        return loss




X_train = X[0:data_num, :]
y_train = Y[0:data_num, :]
X_test = X_out[0:test_num, :]
y_test = Y_out[0:test_num, :]
X_train = Variable((X_train).to(device),requires_grad=True)
y_train = Variable((y_train).to(device))
X_test = Variable((X_test).to(device))
y_test = Variable((y_test).to(device))
Database_meta=data_ini(100)
lamda=[1.,1.,1.,1.,1.]
Net = ANN(2, 50, 4).to(device)
optimizer = torch.optim.Adam(Net.parameters())  # 优化器使用随机梯度下降，传入网络参数和学习率
# ------------------pre_train with data-----------------------

if MODEL=='Measurement':
    print('===========Pretrain=============')
    if os.path.exists(f'model_save/model_save_initial_out.pkl') == False:
        torch.save(Net.state_dict(),
                   f'model_save/model_save_initial_out.pkl')
        torch.save(optimizer.state_dict(),
                   f'model_save/optimizer_save_initial_out.pkl')
    if data_num != 0:
        l_record = 1e8
        for iter in tqdm(range(pretrain_num)):
            optimizer.zero_grad()
            Loss=Loss_func(Database_meta,Net)
            l_data = Loss.loss_data(X_train, y_train)
            l_data.backward()
            optimizer.step()
            if (iter + 1) % 1000 == 0:
                if l_data.cpu().data.numpy() < l_record:
                    torch.save(Net.state_dict(),
                               f'model_save/model_save_pretrain_{data_num}_out.pkl')
                    torch.save(optimizer.state_dict(),
                               f'model_save/optimizer_save_pretrain_{data_num}_out.pkl')
                else:
                    break
                l_record = l_data.cpu().data.numpy()

    print('========Pretrain Finish!=========')

    measure_num = 20000
    MSE_record = np.zeros([2, 2, 2, 2, 2,5])
    for bit_1 in [0, 1]:
        for bit_2 in [0, 1]:
            for bit_3 in [0, 1]:
                for bit_4 in [0, 1]:
                    for bit_5 in [0, 1]:
                        l_record = 1e8
                        if data_num == 0:
                            Net.load_state_dict(
                                torch.load(f'model_save/model_save_initial_out.pkl'))
                            optimizer.load_state_dict(
                                torch.load(f'model_save/optimizer_save_initial_out.pkl'))
                        else:
                            Net.load_state_dict(
                                torch.load(
                                    f'model_save/model_save_pretrain_{data_num}_out.pkl'))
                            optimizer.load_state_dict(
                                torch.load(
                                    f'model_save/optimizer_save_pretrain_{data_num}_out.pkl'))
                        for iter in tqdm(range(measure_num)):
                            optimizer.zero_grad()
                            Loss = Loss_func(Database_meta, Net)
                            reg_control = [bit_1, bit_2, bit_3, bit_4, bit_5]
                            if data_num == 0:
                                if reg_control == [0, 0, 0, 0, 0]:
                                    loss = Variable(torch.tensor([0.]), requires_grad=True)
                                else:
                                    loss = 0
                            else:
                                loss = Loss.loss_data(X_train, y_train)

                            if reg_control[0] == 1:
                                l_c = Loss.loss_c()
                                loss += lamda[0] * l_c

                            if reg_control[1] == 1:
                                l_d = Loss.loss_d()
                                loss += lamda[1] * l_d

                            if reg_control[2] == 1:
                                l_e= Loss.loss_e()
                                loss += lamda[2] * l_e

                            if reg_control[3] == 1:
                                l_f = Loss.loss_f()
                                loss += lamda[3] * l_f

                            if reg_control[4] == 1:
                                l_r= Loss.loss_range()
                                loss += lamda[4] * l_r

                            loss.backward()
                            optimizer.step()
                            if (iter + 1) % 1000 == 0:
                                if loss.cpu().data.numpy() > l_record:
                                    break

                                l_record = loss.cpu().data.numpy()
                                torch.save(Net.state_dict(),
                                           f'model_save/model_save_{data_num}_out.pkl')
                                torch.save(optimizer.state_dict(),
                                           f'model_save/optimizer_save_{data_num}_out.pkl')

                        Net.load_state_dict(
                            torch.load(f'model_save/model_save_{data_num}_out.pkl'))
                        prediction_test = Net(X_test).cpu().data.numpy()
                        c_loss = np.mean((y_test[:, 0].cpu().data.numpy() - prediction_test[:, 0]) ** 2)
                        d_loss = np.mean((y_test[:, 1].cpu().data.numpy() - prediction_test[:, 1]) ** 2)
                        e_loss = np.mean((y_test[:, 2].cpu().data.numpy() - prediction_test[:, 2]) ** 2)
                        f_loss = np.mean((y_test[:, 3].cpu().data.numpy() - prediction_test[:, 3]) ** 2)
                        total_loss = np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2)
                        MSE_record[bit_1, bit_2, bit_3, bit_4, bit_5,0] =total_loss
                        MSE_record[bit_1, bit_2, bit_3, bit_4, bit_5, 1] = c_loss
                        MSE_record[bit_1, bit_2, bit_3, bit_4, bit_5, 2] = d_loss
                        MSE_record[bit_1, bit_2, bit_3, bit_4, bit_5, 3] = e_loss
                        MSE_record[bit_1, bit_2, bit_3, bit_4, bit_5,4] = f_loss
                        print(f'reg:  {reg_control},  loss:   {loss.cpu().data.numpy()},  loss_valid:{total_loss}')
                        time.sleep(0.1)

    np.save(f'result_save/MSE_data_{data_num}_out.npy', MSE_record)
    end_time = time.time()
    print('total time:',end_time-start_time)
if MODEL=='SHAP':
    def PINN_SHAP(result, name, plot_pic=True):
        a = np.log10(np.divide(result[0, :, :, :, :], result[1, :, :, :, :])).reshape(-1, 1)
        b = np.log10(np.divide(result[:, 0, :, :, :], result[:, 1, :, :, :])).reshape(-1, 1)
        c = np.log10(np.divide(result[:, :, 0, :, :], result[:, :, 1, :, :])).reshape(-1, 1)
        d = np.log10(np.divide(result[:, :, :, 0, :], result[:, :, :, 1, :])).reshape(-1, 1)
        e = np.log10(np.divide(result[:, :, :, :, 0], result[:, :, :, :, 1])).reshape(-1, 1)
        a = np.mean(a)
        b = np.mean(b)
        c = np.mean(c)
        d = np.mean(d)
        e = np.mean(e)

        plot_array = [a, b, c, d, e]

        if plot_pic == True:
            plt.figure(4, figsize=(3, 2), dpi=300)
            plt.style.use('ggplot')
            ax = plt.gca()
            ax.invert_yaxis()
            plt.barh(np.arange(0, 4), np.array(plot_array), color='#838AB4')
            # plt.xticks(np.round(np.arange(-0.2,0.8,0.1),1),np.round(np.arange(-0.2,0.8,0.1),1),fontproperties='Arial',size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.yticks(np.arange(0, 4), ['reg1', 'reg2', 'reg3', 'reg4'], fontproperties='Arial', size=7)
            plt.savefig(f'fig_save/bar_{data_num}_{name}.tiff', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/bar_{data_num}_{name}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/bar_{data_num}_{name}.jpg', bbox_inches='tight', dpi=300)
            plt.show()

        return plot_array

    def calcu_SHAP():
        result = np.load(f'result_save/MSE_data_{data_num}_out.npy')
        name=['total','c','d','e','f']
        for i in range(len(name)):
            shap_value = PINN_SHAP(result[:,:,:,:,:,i],name[i], plot_pic=False)
            print(f'{name}:{shap_value}')
        return shap_value

    calcu_SHAP()