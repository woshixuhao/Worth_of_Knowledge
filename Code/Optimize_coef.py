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

start_time=time.time()
torch.manual_seed(525)
np.random.seed(1101)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Optimize the regularization parameter
'''
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

def PINN_SHAP(result,name, plot_pic=False):
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

        return np.array(plot_array)

MODEL='plot'
data_out=np.load('data_save/dataset_out.npy')
data = np.load(f'data_save/dataset.npy')
data_num=100
X=torch.from_numpy(data[:,0:2].astype(np.float32))
Y=torch.from_numpy(data[:,2:].astype(np.float32))
X_out=torch.from_numpy(data_out[:,0:2].astype(np.float32))
Y_out=torch.from_numpy(data_out[:,2:].astype(np.float32))
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
X_out = X_out[0:test_num, :]
y_out = Y_out[0:test_num, :]
X_train = Variable((X_train).to(device),requires_grad=True)
y_train = Variable((y_train).to(device))
X_test = Variable((X_test).to(device))
y_test = Variable((y_test).to(device))
X_out = Variable((X_out).to(device))
y_out = Variable((y_out).to(device))
Database_meta=data_ini(100)
lamda=[1.,1.,1.,1.,1.]
Net = ANN(2, 50, 4).to(device)
optimizer = torch.optim.Adam(Net.parameters())  # 优化器使用随机梯度下降，传入网络参数和学习率
# ------------------pre_train with data-----------------------
adaptive=True
if MODEL=='Optimize':
    mu_coef=1
    aa_ratio = np.array([0., 0., 0., 0.,0.])
    lamda = [1., 1., 1., 1.,1.]
    test_loss = []
    out_loss = []
    coef_save = []
    shap_save = []

    print('===========Pretrain=============')
    if os.path.exists(f'model_save/model_save_initial_opt.pkl') == False:
        torch.save(Net.state_dict(),
                   f'model_save/model_save_initial_opt.pkl')
        torch.save(optimizer.state_dict(),
                   f'model_save/optimizer_save_initial_opt.pkl')
    if data_num != 0:
        l_record = 1e8
        for iter in tqdm(range(pretrain_num)):
            optimizer.zero_grad()
            Loss = Loss_func(Database_meta, Net)
            l_data = Loss.loss_data(X_train, y_train)
            l_data.backward()
            optimizer.step()
            if (iter + 1) % 1000 == 0:
                if l_data.cpu().data.numpy() < l_record:
                    torch.save(Net.state_dict(),
                               f'model_save/model_save_pretrain_{data_num}_opt.pkl')
                    torch.save(optimizer.state_dict(),
                               f'model_save/optimizer_save_pretrain_{data_num}_opt.pkl')
                else:
                    break
                l_record = l_data.cpu().data.numpy()

    print('========Pretrain Finish!=========')

    for opt_iter in range(20):

        measure_num = 20000
        MSE_record = np.zeros([2, 2, 2, 2, 2,5])
        for lamda_i in range(len(lamda)):
            lamda[lamda_i] = (10 ** (mu_coef * aa_ratio[lamda_i])) * lamda[lamda_i]
        print(lamda)
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            l_record = 1e8
                            if data_num == 0:
                                Net.load_state_dict(
                                    torch.load(f'model_save/model_save_initial_opt.pkl'))
                                optimizer.load_state_dict(
                                    torch.load(f'model_save/optimizer_save_initial_opt.pkl'))
                            else:
                                Net.load_state_dict(
                                    torch.load(
                                        f'model_save/model_save_pretrain_{data_num}_opt.pkl'))
                                optimizer.load_state_dict(
                                    torch.load(
                                        f'model_save/optimizer_save_pretrain_{data_num}_opt.pkl'))
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
                                               f'model_save/model_save_{data_num}_opt.pkl')
                                    torch.save(optimizer.state_dict(),
                                               f'model_save/optimizer_save_{data_num}_opt.pkl')

                            Net.load_state_dict(
                                torch.load(f'model_save/model_save_{data_num}_opt.pkl'))
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

        np.save(f'result_save/MSE_data_{data_num}_opt.npy', MSE_record)
        end_time = time.time()
        print('total time:',end_time-start_time)

        if adaptive == True:
            Net.load_state_dict(
                torch.load(f'model_save/model_save_{data_num}_opt.pkl'))
            prediction_test = Net(X_test).cpu().data.numpy()
            prediction_out = Net(X_out).cpu().data.numpy()
            if opt_iter == 0:
                test_loss.append(np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2))
                out_loss.append(np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2))
                print('test_loss', np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2))
                print('out_loss', np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2))
                result = np.load(f'result_save/MSE_data_{data_num}_opt.npy')
                SHAP = PINN_SHAP(result,'good')
                aa_ratio = SHAP
                coef_save.append(np.array(lamda).copy())
                shap_save.append(aa_ratio)
                print('SHAP:', shap_save)
                print('test_loss:', test_loss)
                print('out_loss:',out_loss)
            if opt_iter > 0:
                if np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2) < test_loss[-1]:
                    test_loss.append(np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2))
                    out_loss.append(np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2))
                    print('test_loss', np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2))
                    print('out_loss', np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2))
                    result = np.load(f'result_save/MSE_data_{data_num}_opt.npy')
                    SHAP = PINN_SHAP(result, 'good')
                    aa_ratio = SHAP
                    coef_save.append(np.array(lamda).copy())
                    shap_save.append(aa_ratio)
                    print('SHAP:', SHAP)
                    print('test_loss:', test_loss)
                    print('out_loss:', out_loss)
                else:
                    print('======Get bad==========')
                    aa_ratio = shap_save[-1]
                    lamda = coef_save[-1].tolist()
                    coef_save.append(np.array(lamda).copy())
                    shap_save.append(aa_ratio)
                    test_loss.append(test_loss[-1])
                    out_loss.append(out_loss[-1])
                    mu_coef *= 0.5
    np.save('result_save/coef_save_delta.npy',coef_save,allow_pickle=True)
    np.save('result_save/shap_save_delta.npy',shap_save,allow_pickle=True)
    np.save('result_save/test_loss_delta.npy',test_loss,allow_pickle=True)
    np.save('result_save/out_loss_delta.npy', out_loss, allow_pickle=True)


if MODEL=='SHAP':



    def calcu_SHAP():
        result = np.load(f'result_save/MSE_data_{data_num}.npy')
        name=['total','c','d','e','f']
        for i in range(len(name)):
            shap_value = PINN_SHAP(result[:,:,:,:,:,i],name[i], plot_pic=False)
            print(f'{name[i]}:{shap_value}')
        return shap_value

    calcu_SHAP()

if MODEL=='validate':
    lamda = [1., 1., 1., 1.,1.]
    test_loss = []
    coef_save = []
    shap_save = []

    lamda=np.load('result_save/coef_save_delta.npy',allow_pickle=True)
    SHAP=np.load('result_save/shap_save_delta.npy',allow_pickle=True)
    test_loss=np.load('result_save/test_loss_delta.npy',allow_pickle=True)
    validate_loss=np.load('result_save/out_loss_delta.npy',allow_pickle=True)
    print(lamda)
    print(SHAP)
    print(test_loss)
    print(validate_loss)
    #lamda=lamda[-1]
    lamda = [1., 1., 1., 1., 1.]

    for t in tqdm(range(20000)):
        Loss = Loss_func(Database_meta, Net)
        optimizer.zero_grad()
        prediction = Net(X_train)
        l_data=Loss.loss_data(X_train, y_train)
        l_c = Loss.loss_c()
        l_d = Loss.loss_d()
        l_e = Loss.loss_e()
        l_f = Loss.loss_f()
        l_r = Loss.loss_range()

        loss= l_data+lamda[0] * l_c+lamda[1] * l_d+lamda[2] * l_e+lamda[3] * l_f+lamda[4] * l_r
        loss.backward()
        optimizer.step()

        if t == 1:
            lamda[0] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_c.cpu().data.numpy())))
            lamda[1] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_d.cpu().data.numpy())))
            lamda[2] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_e.cpu().data.numpy())))
            lamda[3] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_f.cpu().data.numpy())))
            lamda[4] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_r.cpu().data.numpy())))
            print(lamda)

    Net.eval()
    prediction_test = Net(X_test).cpu().data.numpy()
    prediction_out = Net(X_out).cpu().data.numpy()
    loss_test = np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2)
    loss_out = np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2)
    print(loss_test)
    print(loss_out)

if MODEL=='Gradient':
    def obtain_loss_grad(u):
        param_index = 0
        for name, params in u.named_parameters():
            if param_index == 0:
                param_grad = torch.clone(params.grad.reshape(-1, 1))
            else:
                param_grad = torch.cat([param_grad, params.grad.reshape(-1, 1)], dim=0)
            param_index += 1
        return param_grad

    for t in tqdm(range(20000)):
        Loss = Loss_func(Database_meta, Net)
        optimizer.zero_grad()
        prediction = Net(X_train)
        l_data=Loss.loss_data(X_train, y_train)
        l_c = Loss.loss_c()
        l_d = Loss.loss_d()
        l_e = Loss.loss_e()
        l_f = Loss.loss_f()
        l_r = Loss.loss_range()

        l_data.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        data_grad_max = torch.max(torch.abs(param_grad))

        l_c.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        reg_grad_1 = torch.mean(torch.abs(param_grad))
        lamda[0] = 0.9 * lamda[0] + 0.1 * (data_grad_max / (reg_grad_1 * lamda[0])).cpu().data.numpy()

        l_d.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        reg_grad_2 = torch.mean(torch.abs(param_grad))
        lamda[1] = 0.9 * lamda[1] + 0.1 * (data_grad_max / (reg_grad_2 * lamda[1])).cpu().data.numpy()

        l_e.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        reg_grad_3 = torch.mean(torch.abs(param_grad))
        lamda[2] = 0.9 * lamda[2] + 0.1 * (data_grad_max / (reg_grad_3 * lamda[2])).cpu().data.numpy()


        l_f.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        reg_grad_4 = torch.mean(torch.abs(param_grad))
        lamda[3] = 0.9 * lamda[3] + 0.1 * (data_grad_max / (reg_grad_4 * lamda[3])).cpu().data.numpy()

        l_r.backward(retain_graph=True)
        param_grad = obtain_loss_grad(Net)
        reg_grad_5 = torch.mean(torch.abs(param_grad))
        lamda[4] = 0.9 * lamda[4] + 0.1 * (data_grad_max / (reg_grad_2 * lamda[4])).cpu().data.numpy()

        loss= l_data+lamda[0] * l_c+lamda[1] * l_d+lamda[2] * l_e+lamda[3] * l_f+lamda[4] * l_r
        loss.backward()
        optimizer.step()

        # if t == 1:
        #     lamda[0] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_c.cpu().data.numpy())))
        #     lamda[1] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_d.cpu().data.numpy())))
        #     lamda[2] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_e.cpu().data.numpy())))
        #     lamda[3] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_f.cpu().data.numpy())))
        #     lamda[4] = 10 ** int((np.log(l_data.cpu().data.numpy() / l_r.cpu().data.numpy())))
        #     print(lamda)

    Net.eval()
    prediction_test = Net(X_test).cpu().data.numpy()
    prediction_out = Net(X_out).cpu().data.numpy()
    loss_test = np.mean((y_test.cpu().data.numpy() - prediction_test) ** 2)
    loss_out = np.mean((y_out.cpu().data.numpy() - prediction_out) ** 2)
    print(loss_test)
    print(loss_out)

if MODEL=='plot':
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 7,
             }
    lamda = np.load('result_save/coef_save_delta.npy', allow_pickle=True)
    SHAP = np.load('result_save/shap_save_delta.npy', allow_pickle=True)
    test_loss = np.load('result_save/test_loss_delta.npy', allow_pickle=True)
    validate_loss = np.load('result_save/out_loss_delta.npy', allow_pickle=True)

    def plot_shap_trend():
        print(SHAP[0])
        plt.figure(1, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        x = np.arange(len(SHAP[0]))  # x轴刻度标签位置
        labels=['Iter=0','Iter=1', 'Iter=4', 'Iter=19']
        width = 0.15  # 柱子的宽度
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
        plt.bar(x - 1.5 * width, SHAP[0], width, label=labels[0])
        plt.bar(x - 0.5 * width, SHAP[1], width, label=labels[1])
        plt.bar(x + 0.5 * width, SHAP[4], width, label=labels[2])
        plt.bar(x + 1.5 * width, SHAP[19], width, label=labels[3])
        plt.xticks(x, ['reg1', 'reg2', 'reg3', 'reg4', 'reg5'], fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/shap.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_loss_trend():
        plt.style.use('ggplot')
        plt.figure(4, figsize=(3.5, 2.5), dpi=300)
        plt.plot(np.squeeze(np.array(test_loss)), marker='x', color='#838AB4', label='valid')
        plt.xticks(fontproperties='Arial', size=8)
        plt.yticks(fontproperties='Arial', size=8)
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.savefig(f'fig_save/loss_trend.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/loss_trend.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/loss_trend.jpg', bbox_inches='tight', dpi=300)
        plt.show()


    def plot_lamda_trend():
        plt.figure(1, figsize=(3.5, 2.5), dpi=300)
        plt.style.use('ggplot')
        x = np.arange(len(SHAP[0]))  # x轴刻度标签位置
        labels = ['Iter=0', 'Iter=1', 'Iter=4', 'Iter=19']
        width = 0.15  # 柱子的宽度
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
        plt.bar(x - 1.5 * width, lamda[0], width, label=labels[0])
        plt.bar(x - 0.5 * width, lamda[1], width, label=labels[1])
        plt.bar(x + 0.5 * width, lamda[4], width, label=labels[2])
        plt.bar(x + 1.5 * width, lamda[19], width, label=labels[3])
        plt.xticks(x, ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5'], fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1)
        plt.savefig(f'fig_save/lamda.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/lamda.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/lamda.jpg', bbox_inches='tight', dpi=300)
        plt.show()


    def plot_total_trend():
        # setup the figure and axes
        fig = plt.figure(1,figsize=(4, 3),dpi=300)
        ax = plt.axes( projection='3d')


        # fake data
        _x = np.arange(5)+0.25
        _y = np.arange(4)+0.25
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = x + y
        print(x)
        print(top)
        l=np.array([lamda[0],lamda[1],lamda[4],lamda[19]]).reshape(-1)
        s = np.array([SHAP[0],SHAP[1], SHAP[4], SHAP[19]]).reshape(-1)
        print(s)
        bottom = np.ones_like(top)
        width = depth = 0.5
        COLOR = ["#8A83B4", "#474554", "#ACA9BB", "#4D9681","#136250"]
        color_list = []
        for i in range(5):
            c = COLOR[i]
            color_list.append([c] * 4)
        color_list = np.asarray(color_list).T

        color_flat=color_list.ravel()
        ax.bar3d(x, y, bottom, width, depth, l-1, shade=True,color=color_flat)


        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
        z1_label = ax.get_zticklabels()
        [z1_label_temp.set_fontname('Arial') for z1_label_temp in z1_label]
        plt.savefig(f'fig_save/lamda_trend.pdf', transparent=True, dpi=300)
        plt.savefig(f'fig_save/lamda_trend.tiff', transparent=True, dpi=300)


        plt.show()

        fig = plt.figure(2, figsize=(4, 3), dpi=300)
        ax = plt.axes(projection='3d')
        bottom = np.zeros_like(top)
        ax.bar3d(x, y, bottom, width, depth, s, shade=True, color=color_flat)
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
        z1_label = ax.get_zticklabels()
        [z1_label_temp.set_fontname('Arial') for z1_label_temp in z1_label]
        plt.savefig(f'fig_save/shap_trend.pdf', transparent=True, dpi=300)
        plt.savefig(f'fig_save/shap_trend.tiff', transparent=True, dpi=300)
        plt.show()

        fig = plt.figure(3, figsize=(3, 2), dpi=300)
        x_loss=[0,1,2,3]
        loss=np.squeeze(np.array(test_loss))
        y_loss=np.array([loss[0],loss[1],loss[4],loss[19]])
        plt.plot(x_loss,y_loss, linestyle='-.', marker='^', markersize=4, color='black', label='KdV equation')

        plt.xticks([],[],fontproperties='Arial', size=8)
        plt.yticks(fontproperties='Arial', size=8)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        plt.savefig(f'fig_save/loss_trend_new.tiff',transparent=True, bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/loss_trend_new.pdf',transparent=True, bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/loss_trend_new.jpg',transparent=True, bbox_inches='tight', dpi=300)
        plt.show()
        plt.show()

    # plot_loss_trend()
    # plot_lamda_trend()
    plot_total_trend()
    #plot_shap_trend()