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
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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


MODEL='SHAP'
data = np.load(f'data_save/dataset.npy')
data_num=0
X=torch.from_numpy(data[:,0:2].astype(np.float32))
Y=torch.from_numpy(data[:,2:].astype(np.float32))
test_num = 5000
pretrain_num=20000

def data_ini(meta_num):
    a_meta = torch.linspace(0, math.pi, meta_num)
    b_meta = torch.linspace(-math.pi, math.pi, meta_num)
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
X_test = X[data.shape[0] - test_num:data.shape[0], :]
y_test = Y[data.shape[0] - test_num:data.shape[0], :]
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
    if os.path.exists(f'model_save/model_save_initial.pkl') == False:
        torch.save(Net.state_dict(),
                   f'model_save/model_save_initial.pkl')
        torch.save(optimizer.state_dict(),
                   f'model_save/optimizer_save_initial.pkl')
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
                               f'model_save/model_save_pretrain_{data_num}.pkl')
                    torch.save(optimizer.state_dict(),
                               f'model_save/optimizer_save_pretrain_{data_num}.pkl')
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
                                torch.load(f'model_save/model_save_initial.pkl'))
                            optimizer.load_state_dict(
                                torch.load(f'model_save/optimizer_save_initial.pkl'))
                        else:
                            Net.load_state_dict(
                                torch.load(
                                    f'model_save/model_save_pretrain_{data_num}.pkl'))
                            optimizer.load_state_dict(
                                torch.load(
                                    f'model_save/optimizer_save_pretrain_{data_num}.pkl'))
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
                                           f'model_save/model_save_{data_num}.pkl')
                                torch.save(optimizer.state_dict(),
                                           f'model_save/optimizer_save_{data_num}.pkl')

                        Net.load_state_dict(
                            torch.load(f'model_save/model_save_{data_num}.pkl'))
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

    np.save(f'result_save/MSE_data_{data_num}.npy', MSE_record)
    end_time = time.time()
    print('total time:',end_time-start_time)


if MODEL=='SHAP':
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 6,
             }
    def PINN_SHAP(result,name, plot_pic=True):
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


    def calcu_SHAP(data_num):
        result = np.load(f'result_save/MSE_data_{data_num}.npy')
        name=['total','c','d','e','f']
        shap=[]
        for i in range(len(name)):
            shap_value = PINN_SHAP(result[:,:,:,:,:,i],name[i], plot_pic=False)
            shap.append(shap_value)
        return np.array(shap)

    def calcu_SHAP_far(data_num):
        result = np.load(f'result_save/MSE_data_{data_num}_out(far).npy')
        name=['total','c','d','e','f']
        shap=[]
        for i in range(len(name)):
            shap_value = PINN_SHAP(result[:,:,:,:,:,i],name[i], plot_pic=False)
            shap.append(shap_value)
        return np.array(shap)

    def calcu_SHAP_out(data_num):
        result = np.load(f'result_save/MSE_data_{data_num}_out.npy')
        name=['total','c','d','e','f']
        shap=[]
        for i in range(len(name)):
            shap_value = PINN_SHAP(result[:,:,:,:,:,i],name[i], plot_pic=False)
            shap.append(shap_value)
        return np.array(shap)

    def plot_SHAP():
        shap_0=calcu_SHAP(0)
        shap_0_out = calcu_SHAP_out(0)
        shap_100 = calcu_SHAP(100)
        shap_100_out = calcu_SHAP_out(100)
        #print(shap_100_out)

        labels=['No data+Interpolation','No data+Extrapolation','100 data+Interpolation','100 data+Extrapolation']
        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        x = np.arange(len(shap_0))  # x轴刻度标签位置
        width = 0.15  # 柱子的宽度
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
        plt.bar(x - 1.5 * width, shap_0[0], width, label=labels[0])
        plt.bar(x - 0.5 * width, shap_0_out[0], width, label=labels[1])
        plt.bar(x + 0.5 * width, shap_100[0], width, label=labels[2])
        plt.bar(x + 1.5 * width, shap_100_out[0], width, label=labels[3])
        plt.xticks(x, ['Rule 1','Rule 2','Rule 3','Rule 4','Rule 5'], fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/shap.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_no_data():
        font1 = {'family': 'Arial',
                 'weight': 'normal',
                 # "style": 'italic',
                 'size': 7,
                 }
        shap_0 = calcu_SHAP(0)
        shap_0_out = calcu_SHAP_out(0)
        shap_100 = calcu_SHAP(100)
        shap_100_out = calcu_SHAP_out(100)
        # print(shap_100_out)

        x = np.arange(len(shap_0))  # the label locations
        width = 0.25  # the width of the bars

        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        ax = plt.axes()
        rects1 = plt.bar(x - width / 2, shap_100[0], width, label='Interpolation', color='#8A83B4')
        rects2 = plt.bar(x + width / 2, shap_100_out[0], width, label='Extrapolation', color='#A1A9D0')
        labels = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5']
        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.xticks(x, labels, fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(5))
        plt.legend(prop=font1)
        plt.ylim(-0.1,1.3)
        plt.savefig(f'fig_save/shap_100_compare.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_100_compare.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_100_compare.jpg', bbox_inches='tight', dpi=300)
        plt.show()


    def plot_contribution():
        shap_0 = calcu_SHAP(0)
        print(shap_0)
        labels = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5']
        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        x = np.arange(len(shap_0))  # x轴刻度标签位置
        width = 0.1  # 柱子的宽度
        COLOR = ["#8A83B4", "#474554", "#ACA9BB", "#4D9681", "#136250"]
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
        plt.bar(x - 2 * width, shap_0[:,0], width, label=labels[0],color=COLOR[0])
        plt.bar(x - 1 * width, shap_0[:,1], width, label=labels[1],color=COLOR[1])
        plt.bar(x, shap_0[:,2], width, label=labels[2],color=COLOR[2])
        plt.bar(x + 1 * width, shap_0[:,3], width, label=labels[3],color=COLOR[3])
        plt.bar(x + 2 * width, shap_0[:,4], width, label=labels[4],color=COLOR[4])
        plt.xticks(x,['Total', 'c', 'd', 'e','f'] , fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/shap_rule.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_rule.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_rule.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_meta_loc():
        shap_100_far = calcu_SHAP_far(100)
        shap_100 = calcu_SHAP(100)
        shap_100_out = calcu_SHAP_out(100)

        labels = ['Interpolation+inner collocation points','Extrapolation+inner collocation points','Extrapolation+outer collocation points']
        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        x = np.arange(len(shap_100))  # x轴刻度标签位置
        width = 0.1  # 柱子的宽度
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
        plt.bar(x - 1 * width, shap_100[0], width, label=labels[0])
        plt.bar(x, shap_100_far[0], width, label=labels[1])
        plt.bar(x + 1 * width, shap_100_out[0], width, label=labels[2])
        plt.xticks(x, ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5'], fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/shap_meta.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_meta.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_meta.jpg', bbox_inches='tight', dpi=300)
        plt.show()
    #plot_SHAP()

    def calcu_rely_boundary_1(data_num):
        if data_num == 'error':
            result = np.load(f'result_save/MSE_data_0(error).npy')[:, :, :, :, :, 0]
        else:
            result = np.load(f'result_save/MSE_data_0.npy')[:, :, :, :, :, 0]
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        reg_control = np.array([bit_1, bit_2, bit_3, bit_4])
                        if np.sum(reg_control) == 0:
                            sum_0.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4]
                                                            , result[1,bit_1,  bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 1:
                            sum_1.append(np.log10(np.divide(result[0,bit_1,  bit_2, bit_3, bit_4]
                                                            , result[1,bit_1,  bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 2:
                            sum_2.append(np.log10(np.divide(result[ 0,bit_1, bit_2, bit_3, bit_4]
                                                            , result[ 1,bit_1, bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 3:
                            sum_3.append(np.log10(np.divide(result[0,bit_1,  bit_2, bit_3, bit_4]
                                                            , result[1,bit_1,  bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 4:
                            sum_4.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4]
                                                            , result[1,bit_1, bit_2, bit_3, bit_4])))

        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.plot(np.arange(0, 5, 1), np.array([mean_1, mean_2, mean_3, mean_4, mean_5]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 5), np.arange(0, 5), fontproperties='Arial', size=7)
        plt.show()

        print(mean_1, mean_2, mean_3, mean_4, mean_5)
        return [mean_1, mean_2, mean_3, mean_4, mean_5]

    def calcu_rely_boundary_2(data_num):
        if data_num == 'error':
            result = np.load(f'result_save/MSE_data_0(error).npy')[:, :, :, :, :, 0]
        else:
            result = np.load(f'result_save/MSE_data_0.npy')[:, :, :, :, :, 0]
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        reg_control = np.array([bit_1, bit_2, bit_3, bit_4])
                        if np.sum(reg_control) == 0:
                            sum_0.append(np.log10(np.divide(result[bit_1, 0, bit_2, bit_3, bit_4]
                                                            , result[bit_1, 1, bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 1:
                            sum_1.append(np.log10(np.divide(result[bit_1, 0, bit_2, bit_3, bit_4]
                                                            , result[bit_1, 1, bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 2:
                            sum_2.append(np.log10(np.divide(result[bit_1, 0, bit_2, bit_3, bit_4]
                                                            , result[bit_1, 1, bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 3:
                            sum_3.append(np.log10(np.divide(result[bit_1, 0, bit_2, bit_3, bit_4]
                                                            , result[bit_1, 1, bit_2, bit_3, bit_4])))
                        if np.sum(reg_control) == 4:
                            sum_4.append(np.log10(np.divide(result[bit_1, 0, bit_2, bit_3, bit_4]
                                                            , result[bit_1, 1, bit_2, bit_3, bit_4])))


        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.plot(np.arange(0, 5, 1), np.array([mean_1, mean_2, mean_3, mean_4, mean_5]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 5), np.arange(0, 5), fontproperties='Arial', size=7)
        plt.show()

        print(mean_1, mean_2, mean_3, mean_4, mean_5)
        return [mean_1, mean_2, mean_3, mean_4, mean_5]


    def calcu_rely_boundary_3(data_num):
        if data_num == 'error':
            result = np.load(f'result_save/MSE_data_0(error).npy')[:, :, :, :, :, 0]
        else:
            result = np.load(f'result_save/MSE_data_0.npy')[:, :, :, :, :, 0]
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        reg_control = np.array([bit_1, bit_2, bit_3, bit_4])
                        if np.sum(reg_control) == 0:
                            sum_0.append(np.log10(np.divide(result[bit_1,  bit_2,0, bit_3, bit_4]
                                                            , result[bit_1,  bit_2,1, bit_3, bit_4])))
                        if np.sum(reg_control) == 1:
                            sum_1.append(np.log10(np.divide(result[bit_1,  bit_2,0, bit_3, bit_4]
                                                            , result[bit_1, bit_2, 1, bit_3, bit_4])))
                        if np.sum(reg_control) == 2:
                            sum_2.append(np.log10(np.divide(result[bit_1,  bit_2,0, bit_3, bit_4]
                                                            , result[bit_1,  bit_2,1, bit_3, bit_4])))
                        if np.sum(reg_control) == 3:
                            sum_3.append(np.log10(np.divide(result[bit_1,  bit_2,0, bit_3, bit_4]
                                                            , result[bit_1,  bit_2,1, bit_3, bit_4])))
                        if np.sum(reg_control) == 4:
                            sum_4.append(np.log10(np.divide(result[bit_1,  bit_2,0, bit_3, bit_4]
                                                            , result[bit_1,  bit_2,1, bit_3, bit_4])))

        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.plot(np.arange(0, 5, 1), np.array([mean_1, mean_2, mean_3, mean_4, mean_5]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 5), np.arange(0, 5), fontproperties='Arial', size=7)
        plt.show()

        print(mean_1, mean_2, mean_3, mean_4, mean_5)
        return [mean_1, mean_2, mean_3, mean_4, mean_5]


    def calcu_rely_boundary_4(data_num):
        if data_num == 'error':
            result = np.load(f'result_save/MSE_data_0(error).npy')[:, :, :, :, :, 0]
        else:
            result = np.load(f'result_save/MSE_data_0.npy')[:, :, :, :, :, 0]
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        reg_control = np.array([bit_1, bit_2, bit_3, bit_4])
                        if np.sum(reg_control) == 0:
                            sum_0.append(np.log10(np.divide(result[bit_1, bit_2,  bit_3,0, bit_4]
                                                            , result[bit_1, bit_2,  bit_3,1, bit_4])))
                        if np.sum(reg_control) == 1:
                            sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, 0, bit_4]
                                                            , result[bit_1, bit_2, bit_3, 1, bit_4])))
                        if np.sum(reg_control) == 2:
                            sum_2.append(np.log10(np.divide(result[bit_1, bit_2,  bit_3,0, bit_4]
                                                            , result[bit_1, bit_2, bit_3, 1, bit_4])))
                        if np.sum(reg_control) == 3:
                            sum_3.append(np.log10(np.divide(result[bit_1, bit_2,  bit_3,0, bit_4]
                                                            , result[bit_1, bit_2, bit_3, 1, bit_4])))
                        if np.sum(reg_control) == 4:
                            sum_4.append(np.log10(np.divide(result[bit_1, bit_2,  bit_3,0, bit_4]
                                                            , result[bit_1, bit_2,  bit_3,1, bit_4])))

        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'))
        plt.plot(np.arange(0, 5, 1), np.array([mean_1, mean_2, mean_3, mean_4, mean_5]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 5), np.arange(0, 5), fontproperties='Arial', size=7)
        plt.show()

        print(mean_1, mean_2, mean_3, mean_4, mean_5)
        return [mean_1, mean_2, mean_3, mean_4, mean_5]


    def calcu_rely_boundary_5(data_num):
        if data_num=='error':
            result = np.load(f'result_save/MSE_data_0(error).npy')[:,:,:,:,:,0]
        else:
            result = np.load(f'result_save/MSE_data_0.npy')[:,:,:,:,:,0]
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        reg_control = np.array([bit_1, bit_2, bit_3, bit_4])
                        if np.sum(reg_control) == 0:
                            sum_0.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4,0]
                                                            , result[bit_1, bit_2, bit_3,  bit_4,1])))
                        if np.sum(reg_control) == 1:
                            sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,  bit_4,0]
                                                            , result[bit_1, bit_2, bit_3, bit_4, 1])))
                        if np.sum(reg_control) == 2:
                            sum_2.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,  bit_4,0]
                                                            , result[bit_1, bit_2, bit_3,  bit_4,1])))
                        if np.sum(reg_control) == 3:
                            sum_3.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,  bit_4,0]
                                                            , result[bit_1, bit_2, bit_3, bit_4, 1])))

                        if np.sum(reg_control) == 4:
                            sum_4.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,  bit_4,0]
                                                            , result[bit_1, bit_2, bit_3,  bit_4,1])))


        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))


        median_1 = np.median(np.array(sum_0))
        median_2 = np.median(np.array(sum_1))
        median_3 = np.median(np.array(sum_2))
        median_4 = np.median(np.array(sum_3))
        median_5 = np.median(np.array(sum_4))
        plt.figure(19,figsize=(4,2),dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0,positions=[0],flierprops=dict(marker='o', markersize=4,linestyle='none'))
        plt.boxplot(sum_1,positions=[1],flierprops=dict(marker='o', markersize=4,linestyle='none'))
        plt.boxplot(sum_2,positions=[2],flierprops=dict(marker='o', markersize=4,linestyle='none'))
        plt.boxplot(sum_3, positions=[3],flierprops=dict(marker='o', markersize=4,linestyle='none'))
        plt.boxplot(sum_4, positions=[4],flierprops=dict(marker='o', markersize=4,linestyle='none'))
        plt.plot(np.arange(0,5,1),np.array([mean_1,mean_2,mean_3,mean_4,mean_5]),c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 5), np.arange(0, 5), fontproperties='Arial', size=7)
        plt.show()

        print(mean_1, mean_2, mean_3, mean_4, mean_5)
        return [mean_1, mean_2, mean_3, mean_4, mean_5]



    def plot_error_true():
        error= np.load(f'result_save/MSE_data_0(error).npy')[:,:,:,:,:,0]
        error=PINN_SHAP(error,'', plot_pic=False)
        true = np.load(f'result_save/MSE_data_0.npy')[:,:,:,:,:,0]
        true = PINN_SHAP(true, '', plot_pic=False)
        x = np.arange(len(error))  # the label locations
        width = 0.35  # the width of the bars

        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        rects1 = plt.bar(x - width / 2, error, width, label='Wrong', color='#8A83B4')
        rects2 = plt.bar(x + width / 2, true, width, label='Correct', color='#A1A9D0')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.xticks(x, ['reg1','reg2','reg3','reg4','reg5'], fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/shap_error_true.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_error_true.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_error_true.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_heatmap():
        plt.figure(1,figsize=(3,3),dpi=300)
        gamma_range = ['Total', 'c', 'd', 'e', 'f']
        alpha_range=['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4','Rule 5']

        plt.xticks(np.arange(len(gamma_range)), alpha_range, rotation=45, fontproperties='Arial', size=9)
        plt.yticks(np.arange(len(gamma_range)), gamma_range, fontproperties='Arial', size=9)
        Corv=calcu_SHAP(0)
        print(Corv)
        for i in range(len(gamma_range)):
            for j in range(len(gamma_range)):
                if np.abs(Corv[i, j]) > 1:
                    text = plt.text(j, i, np.round(Corv[i, j], 2),
                                    ha="center", va="center", color="white", fontproperties='Arial', size=9)
                else:
                    text = plt.text(j, i, np.round(Corv[i, j], 2),
                                    ha="center", va="center", color="black", fontproperties='Arial', size=9)


        clist = ['#FAF8FF', '#6969FF']
        newcmp = LinearSegmentedColormap.from_list('chaos', clist)
        plt.imshow(Corv, interpolation='nearest', cmap=newcmp)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        cb1 = plt.colorbar(fraction=0.045)
        plt.clim(-0.3, 2.5)
        plt.savefig(r'fig_save/plot_corr.png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(r'fig_save/plot_corr.pdf',
                    bbox_inches='tight', dpi=300)
        plt.show()

    # plot_no_data()
    # plot_meta_loc()
    # plot_SHAP()
    # plot_heatmap()
    # plot_contribution()
    a=calcu_rely_boundary_1('e')
    b=calcu_rely_boundary_2('e')
    c=calcu_rely_boundary_3('e')
    d=calcu_rely_boundary_4('e')
    e=calcu_rely_boundary_5('e')

    def plot_rely(a,b,c,d,e):
        plt.figure(7, figsize=(2.8, 2.0), dpi=300)
        plt.style.use('ggplot')
        plt.plot(np.arange(0, 5), np.array(a),linestyle='-.', marker='.', markersize=4, color='#C7483D',label='Rule 1')
        plt.plot(np.arange(0, 5), np.array(b),  linestyle='-.',marker='^',markersize=4,color='#785348',label='Rule 2')
        plt.plot(np.arange(0, 5), np.array(c), linestyle='-.',marker='s',markersize=4,color='#8A83B4',label='Rule 3')
        plt.plot(np.arange(0, 5), np.array(d), linestyle='-.', marker='*', markersize=4, color='#767777',label='Rule 4')
        plt.plot(np.arange(0, 5), np.array(e),linestyle='-.', marker='x',markersize=4, color='#6969FF',label='Rule 5')
        plt.yticks(fontproperties='Arial', size=6)
        labels = [0,1,2,3,4]
        x = np.arange(len(labels))  # the label locations
        plt.xticks(x, labels, fontproperties='Arial', size=6)
        plt.legend(prop=font1,loc='upper left',ncol=2)
        plt.savefig(f'fig_save/rely_0.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rely_0.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rely_0.jpg', bbox_inches='tight', dpi=300)
        plt.show()
    #print(d)
    #
    plot_rely(a,b,c,d,e)
    #plot_error_true()

