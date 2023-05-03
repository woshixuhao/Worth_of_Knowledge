
import torch
import math
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm
from torch.autograd import Variable
import os
import numpy as np
import time
from pyDOE import *
import matplotlib.pyplot as plt
from scipy.stats import uniform
# Domain and Sampling
start=time.time()
torch.manual_seed(525)
np.random.seed(1101)

def interior(n=30):
    x_origin = torch.linspace(0,1,n)
    y_origin = torch.linspace(0,1,n)
    x=torch.zeros([n*n,1])
    y = torch.zeros([n * n, 1])
    num=0
    for i in range(n):
        for j in range(n):
            x[num,0]=x_origin[i]
            y[num,0]=y_origin[j]
            num+=1
    x=x.to(device)
    y=y.to(device)

    cond = (2 - x ** 2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down_yy(n=100):
    x =torch.linspace(0,1,n).reshape(-1,1).to(device)
    y = torch.zeros_like(x).to(device)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up_yy(n=100):
    x = torch.linspace(0,1,n).reshape(-1,1).to(device)
    y = torch.ones_like(x).to(device)
    cond = x ** 2 /math.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down(n=100):
    x = torch.linspace(0,1,n).reshape(-1,1).to(device)
    y = torch.zeros_like(x).to(device)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up(n=100):
    x = torch.linspace(0,1,n).reshape(-1,1).to(device)
    y = torch.ones_like(x).to(device)
    cond = x ** 2 / math.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def left(n=100):
    y = torch.linspace(0,1,n).reshape(-1,1).to(device)
    x = torch.zeros_like(y).to(device)
    cond = torch.zeros_like(x).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def right(n=100):
    y = torch.linspace(0,1,n).reshape(-1,1).to(device)
    x = torch.ones_like(y).to(device)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def make_data(nx,nt):
    num = 0
    data = torch.zeros(2)
    database_ref = torch.zeros([nx * nt, 2])
    x_ref=torch.linspace(0,1,nx)
    y_ref=torch.linspace(0,1,nx)
    u_ref=torch.zeros([nx * nt, 1])
    for j in range(nx):
        for i in range(nt):
            data[0] = x_ref[j]
            data[1] = y_ref[i]
            database_ref[num] = data
            u_ref[num]=x_ref[j]**2*torch.exp(-y_ref[i])
            num += 1
    return database_ref,u_ref
# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)



# Loss
loss = torch.nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def l_interior(u):
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, x, 2) - gradients(uxy, y, 4), cond)


def l_down_yy(u):
    x, y, cond = down_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)


def l_up_yy(u):
    x, y, cond = up_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)


def l_down(u):
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_up(u):
    x, y, cond = up()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_left(u):
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_right(u):
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def PINN_SHAP(result,plot_pic=True):
    a = np.log10(np.divide(result[0, :, :, :, :, :, :], result[1, :, :, :, :, :, :])).reshape(-1, 1)
    b = np.log10(np.divide(result[:, 0, :, :, :, :, :], result[:, 1, :, :, :, :, :])).reshape(-1, 1)
    c = np.log10(np.divide(result[:, :, 0, :, :, :, :], result[:, :, 1, :, :, :, :])).reshape(-1, 1)
    d = np.log10(np.divide(result[:, :, :, 0, :, :, :], result[:, :, :, 1, :, :, :])).reshape(-1, 1)
    e = np.log10(np.divide(result[:, :, :, :, 0, :, :], result[:, :, :, :, 1, :, :])).reshape(-1, 1)
    f = np.log10(np.divide(result[:, :, :, :, :, 0, :], result[:, :, :, :, :, 1, :])).reshape(-1, 1)
    g = np.log10(np.divide(result[:, :, :, :, :, :, 0], result[1, :, :, :, :, :, 1])).reshape(-1, 1)
    a = np.mean(a)
    b = np.mean(b)
    c = np.mean(c)
    d = np.mean(d)
    e = np.mean(e)
    f = np.mean(f)
    g = np.mean(g)
    plot_array = [a, b, c, d, e, f, g]

    if plot_pic==True:
        plt.figure(4, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.barh(np.arange(0, 7), np.array(plot_array), color='#838AB4')
        # plt.xticks(np.round(np.arange(-0.2,0.8,0.1),1),np.round(np.arange(-0.2,0.8,0.1),1),fontproperties='Arial',size=7)
        plt.xticks(fontproperties='Arial', size=7)
        plt.vlines(0, ymin=-0.5, ymax=6.5, color='grey', linewidth=1.5)
        plt.yticks(np.arange(0, 7), ['reg1', 'reg2', 'reg3', 'reg4', 'reg5', 'reg6', 'reg7'], fontproperties='Arial',
                   size=7)
        plt.savefig(f'fig_save/fig.total.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/fig.total.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    return plot_array
try:
    os.makedirs(f'model_save/trail')
except OSError:
    pass

try:
    os.makedirs(f'model_save/pre_train')
except OSError:
    pass
# Training
MODEL='SHAP'
data_num=50
noise_level=0
valid_num=10000
measure_num=20000
database_ref,u_ref=make_data(150,150)

# plt.imshow(u_ref.reshape(150,150).cpu().data.numpy())
# plt.colorbar()
# plt.show()

data_array=np.arange(0,database_ref.shape[0],1)
np.random.shuffle(data_array)
random_index=data_array[0:data_num].tolist()
valid_index=data_array[len(data_array)-valid_num:].tolist()

DR_train=torch.clone(database_ref[random_index])
Y_train=torch.clone(u_ref[random_index])
DR_valid=torch.clone(database_ref[valid_index])
Y_valid=torch.clone(u_ref[valid_index])  #Causion!!


DR_train = Variable(DR_train, requires_grad=True).to(device)
Y_train = Variable(Y_train, requires_grad=True).to(device)
DR_valid = Variable(DR_valid, requires_grad=True).to(device)
Y_valid = Variable(Y_valid, requires_grad=True).to(device)


u = MLP().to(device)
opt = torch.optim.Adam(params=u.parameters())



def obtain_loss_grad(u):
    param_index = 0
    for name, params in u.named_parameters():
        if param_index == 0:
            param_grad = torch.clone(params.grad.reshape(-1, 1))
        else:
            param_grad = torch.cat([param_grad, params.grad.reshape(-1, 1)], dim=0)
        param_index += 1
    return param_grad

if MODEL=='Measurement':
    if data_num == 0:

        torch.save(u.state_dict(), f'model_save/pre_train/model_save_initial.pkl')
        torch.save(opt.state_dict(),
                   f'model_save/pre_train/optimizer_save_initial.pkl')
    else:
        pretrain_num = 20000
        l_record = 1e8
        for iter in tqdm(range(pretrain_num)):
            opt.zero_grad()
            l_data = torch.mean((u(DR_train) - Y_train) ** 2)
            l_data.backward()
            opt.step()
            if (iter + 1) % 1000 == 0:
                if l_data.cpu().data.numpy() < l_record:
                    torch.save(u.state_dict(),
                               f'model_save/pre_train/model_save_pretrain_{data_num}_{noise_level}.pkl')
                    torch.save(opt.state_dict(),
                               f'model_save/pre_train/optimizer_save_pretrain_{data_num}_{noise_level}.pkl')
                else:
                    break
                l_record = l_data.cpu().data.numpy()

    MSE_record_all = np.zeros([2, 2, 2, 2, 2, 2, 2])
    for bit_1 in [0, 1]:
        for bit_2 in [0, 1]:
            for bit_3 in [0, 1]:
                for bit_4 in [0, 1]:
                    for bit_5 in [0, 1]:
                        for bit_6 in [0, 1]:
                            for bit_7 in [0,1]:
                                lamda = [1.,1.,1.,1.,1.,1.,1.]
                                reg_control = [bit_1,bit_2,bit_3,bit_4,bit_5,bit_6,bit_7]
                                l_record = 1e8
                                if data_num == 0:
                                    u.load_state_dict(
                                        torch.load(f'model_save/pre_train/model_save_initial.pkl'))
                                    opt.load_state_dict(
                                        torch.load(f'model_save/pre_train/optimizer_save_initial.pkl'))
                                else:
                                    u.load_state_dict(
                                        torch.load(
                                            f'model_save/pre_train/model_save_pretrain_{data_num}_{noise_level}.pkl'))
                                    opt.load_state_dict(
                                        torch.load(
                                            f'model_save/pre_train/optimizer_save_pretrain_{data_num}_{noise_level}.pkl'))
                                for iter in tqdm(range(measure_num)):
                                    opt.zero_grad()
                                    if data_num==0:
                                        if reg_control == [0, 0, 0, 0, 0, 0, 0]:
                                            l = Variable(torch.tensor([0.]), requires_grad=True)
                                        else:
                                            l = 0
                                    if data_num!=0:
                                        l=torch.mean((u(DR_train)-Y_train)**2)

                                    if reg_control[0] == 1:
                                        l += lamda[0]*l_interior(u)

                                    if reg_control[1] == 1:
                                        l += lamda[1]*l_up_yy(u)

                                        #print(reg_grad_2)

                                    if reg_control[2] == 1:

                                        l += lamda[2]*l_down_yy(u)



                                    if reg_control[3]== 1:

                                        l += lamda[3]*l_up(u)



                                    if reg_control[4]== 1:
                                        l += lamda[4]*l_down(u)

                                        #print(reg_grad_5)
                                    if reg_control[5]== 1:

                                        l += lamda[5]*l_left(u)

                                        #print(reg_grad_6)
                                    if reg_control[6]==1:


                                        l += lamda[6]*l_right(u)


                                    opt.zero_grad()
                                    l.backward()
                                    opt.step()

                                    if (iter + 1) % 1000 == 0:

                                        if l.cpu().data.numpy() > l_record:
                                            break

                                        l_record = l.cpu().data.numpy()
                                        torch.save(u.state_dict(),
                                                   f'model_save/train/model_save_{data_num}_{noise_level}.pkl')
                                        torch.save(opt.state_dict(),
                                                   f'model_save/train/optimizer_save_{data_num}_{noise_level}.pkl')

                                u.load_state_dict(
                                    torch.load(f'model_save/train/model_save_{data_num}_{noise_level}.pkl'))
                                prediction = u(DR_valid).cpu().data.numpy()
                                loss_valid = np.mean((Y_valid.cpu().data.numpy() - prediction) ** 2)
                                MSE_record_all[bit_1,bit_2,bit_3,bit_4,bit_5,bit_6,bit_7]=loss_valid
                                print(f'reg:  {reg_control},  loss:   {l_record},  loss_valid:{loss_valid}')

    np.save(f'result_save/MSE_{data_num}_data_predict',MSE_record_all)
    print(MSE_record_all)
    end=time.time()
    print('total time: ',end-start)


if MODEL=='SHAP':
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 5,
             }
    def calcu_SHAP(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        shap_value = PINN_SHAP(result, plot_pic=False)
        return shap_value

    def calcu_rely_PDE(data_num,plot=True):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0=[]
        sum_1=[]
        sum_2=[]
        sum_3=[]
        sum_4=[]
        sum_5=[]
        sum_6=[]
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control)==0:
                                    sum_0.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==1:
                                    sum_1.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==2:
                                    sum_2.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==3:
                                    sum_3.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==4:
                                    sum_4.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==5:
                                    sum_5.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control)==6:
                                    sum_6.append(np.log10(np.divide(result[0,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[1,bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])))
        mean_1=np.mean(np.array(sum_0))
        mean_2=np.mean(np.array(sum_1))
        mean_3=np.mean(np.array(sum_2))
        mean_4=np.mean(np.array(sum_3))
        mean_5=np.mean(np.array(sum_4))
        mean_6=np.mean(np.array(sum_5))
        mean_7=np.mean(np.array(sum_6))

        median_1=np.median(np.array(sum_0))
        median_2=np.median(np.array(sum_1))
        median_3=np.median(np.array(sum_2))
        median_4=np.median(np.array(sum_3))
        median_5=np.median(np.array(sum_4))
        median_6=np.median(np.array(sum_5))
        median_7=np.median(np.array(sum_6))

        if plot==True:
            plt.figure(19,figsize=(4,2),dpi=300)
            plt.style.use('ggplot')
            plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                        showmeans=False,
                        meanprops=dict(color='#B94E3E', markersize=3))
            # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
            #          linestyle='--')
            plt.plot(np.arange(0, 7, 1),
                     np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E',marker='^',markersize=3,
                     linestyle='--')
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
            plt.savefig(f'fig_save/PDE_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/PDE_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/PDE_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
            plt.show()


        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)
        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1,0, bit_2, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1,1, bit_2, bit_3, bit_4, bit_5, bit_6])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))
        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_1_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_1_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_1_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary_2(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1, bit_2,0, bit_3, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2,1, bit_3, bit_4, bit_5, bit_6])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))

        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_2_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_2_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_2_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)
        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary_3(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1, bit_2, bit_3,0, bit_4, bit_5, bit_6]
                                                                    , result[
                                                                       bit_1, bit_2, bit_3,1, bit_4, bit_5, bit_6])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))
        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)

        median_1 = np.median(np.array(sum_0))
        median_2 = np.median(np.array(sum_1))
        median_3 = np.median(np.array(sum_2))
        median_4 = np.median(np.array(sum_3))
        median_5 = np.median(np.array(sum_4))
        median_6 = np.median(np.array(sum_5))
        median_7 = np.median(np.array(sum_6))

        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_3_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_3_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_3_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_4(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, 0, bit_5, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, 1, bit_5, bit_6])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))
        print(mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7)

        median_1 = np.median(np.array(sum_0))
        median_2 = np.median(np.array(sum_1))
        median_3 = np.median(np.array(sum_2))
        median_4 = np.median(np.array(sum_3))
        median_5 = np.median(np.array(sum_4))
        median_6 = np.median(np.array(sum_5))
        median_7 = np.median(np.array(sum_6))

        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_4_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_4_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_4_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()


        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_5(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, 0, bit_6]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, 1, bit_6])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))
        print(mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7)
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_5_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_5_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_5_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_6(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_predict.npy')
        sum_0 = []
        sum_1 = []
        sum_2 = []
        sum_3 = []
        sum_4 = []
        sum_5 = []
        sum_6 = []
        for bit_1 in [0, 1]:
            for bit_2 in [0, 1]:
                for bit_3 in [0, 1]:
                    for bit_4 in [0, 1]:
                        for bit_5 in [0, 1]:
                            for bit_6 in [0, 1]:
                                reg_control = np.array([bit_1, bit_2, bit_3, bit_4, bit_5, bit_6])
                                if np.sum(reg_control) == 0:
                                    sum_0.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 1:
                                    sum_1.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 2:
                                    sum_2.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 3:
                                    sum_3.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 4:
                                    sum_4.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 5:
                                    sum_5.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
                                if np.sum(reg_control) == 6:
                                    sum_6.append(np.log10(np.divide(result[bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 0]
                                                                    , result[
                                                                        bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, 1])))
        mean_1 = np.mean(np.array(sum_0))
        mean_2 = np.mean(np.array(sum_1))
        mean_3 = np.mean(np.array(sum_2))
        mean_4 = np.mean(np.array(sum_3))
        mean_5 = np.mean(np.array(sum_4))
        mean_6 = np.mean(np.array(sum_5))
        mean_7 = np.mean(np.array(sum_6))
        print(mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7)
        plt.figure(19, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'),
                    showmeans=False,
                    meanprops=dict(color='#B94E3E', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#B94E3E', marker='^',
                 markersize=3,
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        plt.savefig(f'fig_save/rule_6_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_6_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rule_6_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()


        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def plot(style='nature'):
        if style=='nature':
            font1 = {'family': 'Arial',
                     'weight': 'normal',
                     "style": 'italic',
                     'size': 7,
                     }
            labels = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', 'Rule 6', 'Rule 7']
            in_means = np.array(
                [0.09129697136707288, 0.3723013633454025, 0.3416216807855599, 1.5505967746312816, 2.1212071696914245,
                 0.979699014659279, 1.2539288949958107])
            # out_means = np.array([0.12839842741527646, 0.37448506480054367, 0.41002500882427373, 1.6068700968814236, 2.111868743111276, 0.943143950296378, 1.1069440040831497])
            in_only = np.array(
                [1.6231683456853332, 0.6195329291101015, 2.0901883930259966, 0.9441174028340183, 2.964498933987568,
                 0.49857205215749195, 0.7189507501231271])
            # out_only=np.array([1.6684152534510426,0.9603455001826617, 2.4108014060993646,1.0014886110765828,3.2588723229165426,-0.030611196391474027,0.09358480536593768])

            # labels = ['Rule 5', 'Rule 3', 'Rule 1', 'Rule 2', 'Rule 4', 'Rule 7', 'Rule 6']
            # in_means = [in_means[5],in_means[3],in_means[1],in_means[2],in_means[4],in_means[7],in_means[6]]
            # out_means = [out_means[5],out_means[3],out_means[1],out_means[2],out_means[4],out_means[7],out_means[6]]
            # in_only=[in_only[5],in_only[3],in_only[1],in_only[2],in_only[4],in_only[7],in_only[6]]
            # out_only=[out_only[5],out_only[3],out_only[1],out_only[2],out_only[4],out_only[7],out_only[6]]

            in_means = [-0.8278530624078478, 0.38147723316456505, 0.3079066815771708, 0.09493456815900861,
                        0.3289183527516415,
                        -0.07337857484522736, -0.37096638460478626]
            # out_means=[-0.43774748722272705, 0.28336515955760666, 0.5886877839000801, -0.06265704344372829, 0.5393473134797948, 0.09850421124267453, -0.04448740620728058]
            in_only = np.array(
                [0.9660017814944797, 0.8965425723502517, 1.6097988226779765, 1.1191332059526413, 1.9897776979275374,
                 0.4657072098013312, 0.5632574070818486])
            # out_only = np.array(
            #     [0.46400443888528636,  0.7784278139156638, 1.976662812150443,  -0.7283173883007049, 0.8035470237737392,
            #      -1.3930782093196488, 0.589316232155502])

            x = np.arange(len(labels))  # the label locations
            width = 0.3  # the width of the bars

            plt.figure(1, figsize=(4, 2), dpi=300)
            plt.style.use('ggplot')
            ax = plt.axes()
            rects1 = plt.bar(x - + width / 2, in_means, width, label='RI', color='#6969FF')
            rects2 = plt.bar(x + width / 2, in_only, width, label='FI', color='#B94E3E')
            # rects2 = plt.bar(x + width / 2, out_means, width, label='Extrapolation',color='#8A83B4')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            # plt.plot(x, in_only,  linestyle='-.', marker='.', markersize=4, color='grey')
            plt.xticks(x, labels, fontproperties='Arial', size=7)
            plt.yticks(fontproperties='Arial', size=7)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(5))
            plt.legend(prop=font1)
            plt.ylim(-1, 3.5)

            # plt.plot(x, out_only, label='UPTIME', color='#3D5488', linewidth=1, linestyle='--', marker='o',markersize=3)

            # Add some text for labels, title and custom x-axis tick labels, etc.

            plt.savefig(f'fig_save/shap_50(nature).tiff', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/shap_50(nature).pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/shap_50(nature).jpg', bbox_inches='tight', dpi=300)
            plt.show()

        else:
            font1 = {'family': 'Arial',
                     'weight': 'normal',
                     "style": 'italic',
                     'size': 7,
                     }
            labels = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', 'Rule 6', 'Rule 7']
            in_means = np.array([0.09129697136707288, 0.3723013633454025, 0.3416216807855599, 1.5505967746312816, 2.1212071696914245, 0.979699014659279, 1.2539288949958107])
            #out_means = np.array([0.12839842741527646, 0.37448506480054367, 0.41002500882427373, 1.6068700968814236, 2.111868743111276, 0.943143950296378, 1.1069440040831497])
            in_only=np.array([1.6231683456853332,0.6195329291101015,2.0901883930259966,0.9441174028340183,2.964498933987568,0.49857205215749195,0.7189507501231271])
            #out_only=np.array([1.6684152534510426,0.9603455001826617, 2.4108014060993646,1.0014886110765828,3.2588723229165426,-0.030611196391474027,0.09358480536593768])


            # labels = ['Rule 5', 'Rule 3', 'Rule 1', 'Rule 2', 'Rule 4', 'Rule 7', 'Rule 6']
            # in_means = [in_means[5],in_means[3],in_means[1],in_means[2],in_means[4],in_means[7],in_means[6]]
            # out_means = [out_means[5],out_means[3],out_means[1],out_means[2],out_means[4],out_means[7],out_means[6]]
            # in_only=[in_only[5],in_only[3],in_only[1],in_only[2],in_only[4],in_only[7],in_only[6]]
            # out_only=[out_only[5],out_only[3],out_only[1],out_only[2],out_only[4],out_only[7],out_only[6]]

            in_means =[-0.8278530624078478, 0.38147723316456505, 0.3079066815771708, 0.09493456815900861, 0.3289183527516415,
             -0.07337857484522736, -0.37096638460478626]
            # out_means=[-0.43774748722272705, 0.28336515955760666, 0.5886877839000801, -0.06265704344372829, 0.5393473134797948, 0.09850421124267453, -0.04448740620728058]
            in_only = np.array(
                [0.9660017814944797, 0.8965425723502517,  1.6097988226779765,  1.1191332059526413,1.9897776979275374,
                 0.4657072098013312, 0.5632574070818486])
            # out_only = np.array(
            #     [0.46400443888528636,  0.7784278139156638, 1.976662812150443,  -0.7283173883007049, 0.8035470237737392,
            #      -1.3930782093196488, 0.589316232155502])

            x = np.arange(len(labels))  # the label locations
            width = 0.3  # the width of the bars

            plt.figure(1,figsize=(4,2),dpi=300)
            plt.style.use('ggplot')
            ax=plt.axes()
            rects1 = plt.bar(x-+ width / 2, in_means, width, label='RI',color='#6969FF')
            rects2 = plt.bar(x + width / 2, in_only, width, label='FI', color='#B94E3E')
            #rects2 = plt.bar(x + width / 2, out_means, width, label='Extrapolation',color='#8A83B4')
            # Add some text for labels, title and custom x-axis tick labels, etc.
            #plt.plot(x, in_only,  linestyle='-.', marker='.', markersize=4, color='grey')
            plt.xticks(x, labels, fontproperties='Arial', size=7)
            plt.yticks(fontproperties='Arial', size=7)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(5))
            plt.legend(prop=font1)
            plt.ylim(-1,3.5)



            #plt.plot(x, out_only, label='UPTIME', color='#3D5488', linewidth=1, linestyle='--', marker='o',markersize=3)

            # Add some text for labels, title and custom x-axis tick labels, etc.

            plt.savefig(f'fig_save/shap_50.tiff', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/shap_50.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'fig_save/shap_50.jpg', bbox_inches='tight', dpi=300)
            plt.show()

    def plot_in_out():
        font1 = {'family': 'Arial',
                 'weight': 'normal',
                 #"style": 'italic',
                 'size': 7,
                 }
        labels = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', 'Rule 6', 'Rule 7']
        in_means = np.array([0.09129697136707288, 0.3723013633454025, 0.3416216807855599, 1.5505967746312816, 2.1212071696914245, 0.979699014659279, 1.2539288949958107])
        out_means = np.array([0.12839842741527646, 0.37448506480054367, 0.41002500882427373, 1.6068700968814236, 2.111868743111276, 0.943143950296378, 1.1069440040831497])



        in_means =[-0.8278530624078478, 0.38147723316456505, 0.3079066815771708, 0.09493456815900861, 0.3289183527516415,
         -0.07337857484522736, -0.37096638460478626]
        out_means=[-0.43774748722272705, 0.28336515955760666, 0.5886877839000801, -0.06265704344372829, 0.5393473134797948, 0.09850421124267453, -0.04448740620728058]


        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars

        plt.figure(1,figsize=(4,2),dpi=300)
        plt.style.use('ggplot')
        ax=plt.axes()
        rects1 = plt.bar(x-+ width / 2, in_means, width, label='Interpolation', color='#8A83B4')
        rects2 = plt.bar(x + width / 2, out_means, width, label='Extrapolation', color='#A1A9D0')
        #rects2 = plt.bar(x + width / 2, out_means, width, label='Extrapolation',color='#8A83B4')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        #plt.plot(x, in_only,  linestyle='-.', marker='.', markersize=4, color='grey')
        plt.xticks(x, labels, fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(5))
        plt.legend(prop=font1)
        #plt.ylim(-1,3.5)



        #plt.plot(x, out_only, label='UPTIME', color='#3D5488', linewidth=1, linestyle='--', marker='o',markersize=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.

        plt.savefig(f'fig_save/shap_50_in_out.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_50_in_out.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/shap_50_in_out.jpg', bbox_inches='tight', dpi=300)
        plt.show()


    def plot_rely(a,b,c,d,e,f,g):
        plt.figure(7, figsize=(4, 2), dpi=300)
        plt.style.use('ggplot')
        plt.plot(np.arange(0, 7), np.array(a), marker='x', color='#2878b5',label='reg1')
        plt.plot(np.arange(0, 7), np.array(b), marker='x', color='#838AB4',label='reg2')
        plt.plot(np.arange(0, 7), np.array(c), marker='x', color='#A8AABC',label='reg3')
        plt.plot(np.arange(0, 7), np.array(d), marker='x', color='#c82423',label='reg4')
        plt.plot(np.arange(0, 7), np.array(e), marker='x', color='#ff8884',label='reg5')
        plt.plot(np.arange(0, 7), np.array(f), marker='x', color='#B77D81',label='reg6')
        plt.plot(np.arange(0, 7), np.array(g), marker='x', color='#338682',label='reg7')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial',size=7)
        plt.legend(prop=font1,loc='upper left',ncol=2)
        plt.ylim([-2,3.5])
        plt.savefig(f'fig_save/rely_50.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rely_50.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/rely_50.jpg', bbox_inches='tight', dpi=300)
        plt.show()
    plot_in_out()
    plot()
    a=calcu_rely_PDE(0)
    b=calcu_rely_boundary(0)
    c=calcu_rely_boundary_2(0)
    d=calcu_rely_boundary_3(0)
    e=calcu_rely_boundary_4(0)
    f=calcu_rely_boundary_5(0)
    g=calcu_rely_boundary_6(0)
    #plot_rely(a,b,c,d,e,f,g)
    #
    # for data in [0,50,100]:
    #     print(calcu_SHAP(data))

if MODEL=='Test':
    plt.imshow(u_ref.reshape(150,150), cmap='coolwarm')
    plt.axis('off')
    plt.colorbar()
    plt.show()