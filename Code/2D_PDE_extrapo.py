
import torch
import math
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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
    data = torch.zeros([1,2])
    x_ref=torch.linspace(0,1,nx)
    y_ref=torch.linspace(0,1,nx)
    num=0
    for j in range(nx):
        for i in range(nt):
            data[0,0] = x_ref[j]
            data[0,1] = y_ref[i]

            if data[0,0]<=0.8 and data[0,0]>=0.2 and data[0,1]<=0.8 and data[0,1]>=0.2:
                continue
            else:
                if num == 0:
                    database_ref = data
                    u_ref = (x_ref[j] ** 2 * torch.exp(-y_ref[i])).reshape(-1, 1)
                else:
                    database_ref= torch.cat([database_ref,data],dim=0)
                    u_ref=torch.cat([u_ref,(x_ref[j]**2*torch.exp(-y_ref[i])).reshape(-1,1)],dim=0)
            num+=1


    return database_ref,u_ref

def make_valid(nx,nt):
    num = 0
    data = torch.zeros(2)
    database_ref = torch.zeros([nx * nt, 2])
    x_ref = torch.linspace(0.2, 0.8, nx)
    y_ref = torch.linspace(0.2, 0.8, nx)
    u_ref = torch.zeros([nx * nt, 1])
    for j in range(nx):
        for i in range(nt):
            data[0] = x_ref[j]
            data[1] = y_ref[i]
            database_ref[num] = data
            u_ref[num] = x_ref[j] ** 2 * torch.exp(-y_ref[i])
            num += 1
    return database_ref, u_ref

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
# Training
MODEL='SHAP'
data_num=50
noise_level=0
valid_num=10000
measure_num=20000
database_ref,u_ref=make_data(150,150)
database_val,u_val=make_valid(100,100)
print(u_ref.shape)
# plt.imshow(u_val.reshape(100,100).cpu().data.numpy())
# plt.colorbar()
# plt.show()

data_array=np.arange(0,database_ref.shape[0],1)
data_array_val=np.arange(0,database_val.shape[0],1)
np.random.shuffle(data_array)
np.random.shuffle(data_array_val)
random_index=data_array[0:data_num].tolist()
valid_index=data_array_val[0:valid_num].tolist()



DR_train=torch.clone(database_ref[random_index])
Y_train=torch.clone(u_ref[random_index])
DR_valid=torch.clone(database_val[valid_index])
Y_valid=torch.clone(u_val[valid_index])  #Causion!!


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

        torch.save(u.state_dict(), f'model_save/pre_train_out/model_save_initial.pkl')
        torch.save(opt.state_dict(),
                   f'model_save/pre_train_out/optimizer_save_initial.pkl')
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
                               f'model_save/pre_train_out/model_save_pretrain_{data_num}_{noise_level}.pkl')
                    torch.save(opt.state_dict(),
                               f'model_save/pre_train_out/optimizer_save_pretrain_{data_num}_{noise_level}.pkl')
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
                                        torch.load(f'model_save/pre_train_out/model_save_initial.pkl'))
                                    opt.load_state_dict(
                                        torch.load(f'model_save/pre_train_out/optimizer_save_initial.pkl'))
                                else:
                                    u.load_state_dict(
                                        torch.load(
                                            f'model_save/pre_train_out/model_save_pretrain_{data_num}_{noise_level}.pkl'))
                                    opt.load_state_dict(
                                        torch.load(
                                            f'model_save/pre_train_out/optimizer_save_pretrain_{data_num}_{noise_level}.pkl'))
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
                                                   f'model_save/train_out/model_save_{data_num}_{noise_level}.pkl')
                                        torch.save(opt.state_dict(),
                                                   f'model_save/train_out/optimizer_save_{data_num}_{noise_level}.pkl')

                                u.load_state_dict(
                                    torch.load(f'model_save/train_out/model_save_{data_num}_{noise_level}.pkl'))
                                prediction = u(DR_valid).cpu().data.numpy()
                                loss_valid = np.mean((Y_valid.cpu().data.numpy() - prediction) ** 2)
                                MSE_record_all[bit_1,bit_2,bit_3,bit_4,bit_5,bit_6,bit_7]=loss_valid
                                print(f'reg:  {reg_control},  loss:   {l_record},  loss_valid:{loss_valid}')

    np.save(f'result_save/MSE_{data_num}_data_out',MSE_record_all)
    print(MSE_record_all)
    end=time.time()
    print('total time: ',end-start)

if MODEL=='SHAP':
    def calcu_SHAP(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
        shap_value = PINN_SHAP(result, plot_pic=False)
        return shap_value

    def calcu_rely_PDE(data_num,plot=True):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
            plt.boxplot(sum_0,positions=[0],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_1,positions=[1],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_2,positions=[2],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_3, positions=[3],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_4, positions=[4],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_5, positions=[5],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            plt.boxplot(sum_6, positions=[6],flierprops=dict(marker='o', markersize=4,linestyle='none'),showmeans=True,meanprops=dict(color='#8A83B4',markersize=3))
            # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
            #          linestyle='--')
            plt.plot(np.arange(0, 7, 1),
                     np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]),c='#8A83B4',
                     linestyle='--')
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
            # plt.savefig(f'fig_save/PDE_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
            # plt.savefig(f'fig_save/PDE_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
            # plt.savefig(f'fig_save/PDE_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
            plt.show()


        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)
        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_1_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_1_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_1_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary_2(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_2_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_2_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_2_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        print(mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7)
        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]

    def calcu_rely_boundary_3(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_3_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_3_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_3_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_4(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_4_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_4_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_4_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()


        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_5(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_5_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_5_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_5_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    def calcu_rely_boundary_6(data_num):
        result = np.load(f'result_save/MSE_{data_num}_data_out.npy')
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
        plt.boxplot(sum_0, positions=[0], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_1, positions=[1], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_2, positions=[2], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_3, positions=[3], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_4, positions=[4], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_5, positions=[5], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        plt.boxplot(sum_6, positions=[6], flierprops=dict(marker='o', markersize=4, linestyle='none'), showmeans=True,
                    meanprops=dict(color='#8A83B4', markersize=3))
        # plt.plot(np.arange(0,7,1),np.array([median_1,median_2,median_3,median_4,median_5,median_6,median_7]),c='#8A83B4',
        #          linestyle='--')
        plt.plot(np.arange(0, 7, 1),
                 np.array([mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7]), c='#8A83B4',
                 linestyle='--')
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(np.arange(0, 7), np.arange(0, 7), fontproperties='Arial', size=7)
        # plt.savefig(f'fig_save/rule_6_rely_{data_num}.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_6_rely_{data_num}.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/rule_6_rely_{data_num}.pdf', bbox_inches='tight', dpi=300)
        plt.show()


        return [mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7]


    a = calcu_rely_PDE(50)
    b = calcu_rely_boundary(50)
    c = calcu_rely_boundary_2(50)
    d = calcu_rely_boundary_3(50)
    e = calcu_rely_boundary_4(50)
    f = calcu_rely_boundary_5(50)
    g = calcu_rely_boundary_6(50)

    for data in [0,50,100]:
        print(calcu_SHAP(data))