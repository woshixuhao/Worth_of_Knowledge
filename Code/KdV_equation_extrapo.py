import torch
import numpy as np
import numpy as np
import torch
from torch.autograd import Variable
import math
from neural_network import  *
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io as scio
from torch.optim import lr_scheduler
import heapq
import time
#GPU set
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')
torch.manual_seed(525)
np.random.seed(1101)
valid_num=10000
x_low=-1
x_up=1
t_low=0
t_up=1
x_num=200
t_num=200
data_num=100
noise_level=50
Activation_function='Sin'
MODEL='Measurement'#['Measurement','Train','SHAP','Test']
Equation_name=f'KdV_equation_{x_num}_out'
lamda=[1,1,1]

print(data_num,noise_level)
try:
    os.makedirs(f'model_save/{Equation_name}/')
except OSError:
    pass

try:
    os.makedirs(f'result_save/{Equation_name}/')
except OSError:
    pass

try:
    os.makedirs(f'fig_save/{Equation_name}/')
except OSError:
    pass

#====referenced data==========
data_path = f'Data/KdV-PINN.mat'
data_ref = scio.loadmat(data_path)
un_ref = data_ref.get("uu")
x_ref = np.squeeze(data_ref.get("x"))
t_ref = np.squeeze(data_ref.get("tt")).reshape(201)
nx_ref=x_ref.shape[0]
nt_ref=t_ref.shape[0]
un_noise=un_ref+noise_level/100*np.std(un_ref)*np.random.randn(*un_ref.shape)

def interior():
    x_inter = torch.linspace(x_low, x_up, x_num)
    t_inter = torch.linspace(t_low, t_up, t_num)
    num = 0
    data = torch.zeros(2)
    database_inter = torch.zeros([x_num*t_num, 2])
    for j in range(x_num):
        for i in range(t_num):
            data[0] = x_inter[j]
            data[1] = t_inter[i]
            database_inter[num] = data
            num += 1
    database_inter = Variable(database_inter, requires_grad=True).to(DEVICE)
    return database_inter

def initial():
    x_ini = torch.linspace(x_low, x_up, x_num).reshape(-1,1)
    t_ini = torch.ones_like(x_ini)*t_low
    database_ini=torch.cat([x_ini,t_ini],dim=1)
    ref_ini =torch.cos(math.pi*x_ini)
    database_ini = Variable(database_ini, requires_grad=True).to(DEVICE)
    ref_ini  = Variable(ref_ini, requires_grad=True).to(DEVICE)
    return database_ini,ref_ini

def boundary_left():
    t_bl = torch.linspace(t_low, t_up, t_num).reshape(-1,1)
    x_bl = torch.ones_like(t_bl) * x_low
    database_bl = torch.cat([x_bl, t_bl], dim=1)
    ref_bl =torch.zeros_like(t_bl)
    database_bl = Variable(database_bl, requires_grad=True).to(DEVICE)
    ref_bl = Variable(ref_bl, requires_grad=True).to(DEVICE)
    return database_bl, ref_bl

def boundary_right():
    t_br = torch.linspace(t_low, t_up, t_num).reshape(-1,1)
    x_br = torch.ones_like(t_br) * x_up
    database_br = torch.cat([x_br, t_br], dim=1)
    ref_br =torch.zeros_like(t_br)
    database_br = Variable(database_br, requires_grad=True).to(DEVICE)
    ref_br = Variable(ref_br, requires_grad=True).to(DEVICE)
    return database_br, ref_br

def dataset_ref():
    nx=x_ref.shape[0]
    nt=int(t_ref.shape[0]/2)
    num = 0
    data = torch.zeros(2)
    database_ref = torch.zeros([nx * nt, 2])
    y_ref=torch.zeros([nx * nt, 1])
    for j in range(nx):
        for i in range(nt):
            data[0] = x_ref[j]
            data[1] = t_ref[i]
            database_ref[num] = data
            y_ref[num]=un_noise[j,i]
            num += 1
    return database_ref,y_ref

def dataset_val():
    nx=x_ref.shape[0]
    nt=int(t_ref.shape[0]/2)
    num = 0
    data = torch.zeros(2)
    database_val = torch.zeros([nx * nt, 2])
    y_val=torch.zeros([nx * nt, 1])
    for j in range(nx):
        for i in range(nt):
            data[0] = x_ref[j]
            data[1] = t_ref[i+int(t_ref.shape[0]/2)]
            database_val[num] = data
            y_val[num]=un_noise[j,i+int(t_ref.shape[0]/2)]
            num += 1
    return database_val,y_val

def dataset_origin():
    nx=x_ref.shape[0]
    nt=t_ref.shape[0]
    num = 0
    data = torch.zeros(2)
    database_ref = torch.zeros([nx * nt, 2])
    y_ref=torch.zeros([nx * nt, 1])
    for j in range(nx):
        for i in range(nt):
            data[0] = x_ref[j]
            data[1] = t_ref[i]
            database_ref[num] = data
            y_ref[num]=un_noise[j,i]
            num += 1
    return database_ref,y_ref



class Loss_func():
    def __init__(self,Net):
        self.Net=Net
    def loss_inter(self,database_inter):
        prediction = Net(database_inter)
        H=prediction.reshape(-1,1)
        H_grad = torch.autograd.grad(outputs=prediction.sum(), inputs=database_inter, create_graph=True)[0]
        Hx = H_grad[:, 0].reshape(-1, 1)
        Ht = H_grad[:, 1].reshape(-1, 1)
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database_inter, create_graph=True)[0][:, 0].reshape(-1, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database_inter, create_graph=True)[0][:, 0].reshape(-1, 1)
        loss=torch.mean((Ht+H*Hx+0.0025*Hxxx)**2)
        return loss

    def loss_ini(self,database_ini,ref_ini):
        prediction = Net(database_ini)
        loss = torch.mean((prediction-ref_ini)**2)
        return loss

    def loss_b(self, database_bl, database_br):
        prediction_bl = Net(database_bl)
        prediction_br = Net(database_br)
        loss = torch.mean((prediction_bl-prediction_br)**2)
        return loss


    def loss_data(self,database,y):
        prediction=Net(database)
        loss=torch.mean((prediction-y)**2)
        return loss

def PINN_SHAP(result,plot_pic=True):
    a = np.log10(np.divide(result[0, :, :], result[1, :, :])).reshape(-1, 1)
    b = np.log10(np.divide(result[:, 0, :], result[:, 1, :])).reshape(-1, 1)
    c = np.log10(np.divide(result[:, :, 0], result[:, :, 1])).reshape(-1, 1)
    a = np.mean(a)
    b = np.mean(b)
    c = np.mean(c)

    plot_array=[a,b,c]

    if plot_pic==True:
        plt.figure(4,figsize=(3,2),dpi=300)
        plt.style.use('ggplot')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.barh(np.arange(0,len(plot_array)),np.array(plot_array),color='#838AB4')
        #plt.xticks(np.round(np.arange(-0.2,0.8,0.1),1),np.round(np.arange(-0.2,0.8,0.1),1),fontproperties='Arial',size=7)
        plt.xticks(fontproperties='Arial',size=7)
        plt.yticks(np.arange(0,len(plot_array)),['reg1','reg2','reg3'],fontproperties='Arial',size=7)
        plt.savefig(f'fig_save/{Equation_name}/bar_{data_num}_{noise_level}.tiff',bbox_inches='tight',dpi=300)
        plt.savefig(f'fig_save/{Equation_name}/bar_{data_num}_{noise_level}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/{Equation_name}/bar_{data_num}_{noise_level}.jpg',bbox_inches='tight',dpi=300)
        plt.show()

    return plot_array

Net=NN(Num_Hidden_Layers=5,
    Neurons_Per_Layer=50,
    Input_Dim=2,
    Output_Dim=1,
    Data_Type=torch.float32,
    Device=DEVICE,
    Activation_Function=Activation_function,
    Batch_Norm=False)
optimizer = torch.optim.Adam([
    {'params': Net.parameters()},
])


database_interior=interior()
database_ini,ref_ini=initial()
database_bl, ref_bl=boundary_left()
database_br,ref_br=boundary_right()
database_ref,y_ref=dataset_ref()
database_val,y_val=dataset_val()
database_ori,y_ori=dataset_origin()
data_array=np.arange(0,database_ref.shape[0],1)
data_array_val=np.arange(0,database_val.shape[0],1)
np.random.shuffle(data_array)
np.random.shuffle(data_array_val)
random_index=data_array[0:data_num].tolist()
valid_index=data_array_val[0:valid_num].tolist()

DR_train=torch.clone(database_ref[random_index])
Y_train=torch.clone(y_ref[random_index])
DR_valid=torch.clone(database_val[valid_index])
Y_valid=torch.clone(y_val[valid_index])  #Causion!!

database_ref = Variable(database_ref, requires_grad=True).to(DEVICE)
y_ref = Variable(y_ref, requires_grad=True).to(DEVICE)
DR_train = Variable(DR_train, requires_grad=True).to(DEVICE)
Y_train = Variable(Y_train, requires_grad=True).to(DEVICE)
DR_valid = Variable(DR_valid, requires_grad=True).to(DEVICE)
Y_valid = Variable(Y_valid, requires_grad=True).to(DEVICE)
database_ori= Variable(database_ori, requires_grad=True).to(DEVICE)
#NN
Loss = Loss_func(Net)
iter_num=20000
pretrain_num=10000
if MODEL=='Train':
    for iter in tqdm(range(iter_num)):
        optimizer.zero_grad()
        l_inter=Loss.loss_inter(database_interior)
        l_ini=Loss.loss_ini(database_ini,ref_ini)
        l_b=Loss.loss_b(database_bl,database_br)
        l_data=Loss.loss_data(DR_train,Y_train)
        if data_num!=0:
            loss=l_inter+l_ini+l_b+l_data
        if data_num==0:
            loss = l_inter + l_ini + l_b
        loss.backward()
        optimizer.step()
        if (iter + 1) % 1000 == 0:
            prediction = Net(database_ref).reshape(nx_ref, nt_ref).cpu().data.numpy()
            loss_valid=np.mean((un_ref - prediction) ** 2)
            print(f'loss:   {loss.cpu().data.numpy()}, loss_valid:{loss_valid}')
    torch.save(Net.state_dict(),
               f'model_save/{Equation_name}/model_save.pkl')

if MODEL=='Test':
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/model_save.pkl'))

    prediction=Net(database_ref).reshape(nx_ref,nt_ref).cpu().data.numpy()
    print(np.mean((un_ref-prediction)**2))


    plt.figure(1,figsize=(6,2),dpi=300)
    plt.subplot(1,3,1)
    plt.imshow(prediction,cmap='coolwarm')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(un_ref,cmap='coolwarm')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(un_ref-prediction,cmap='coolwarm')
    plt.colorbar()
    plt.show()

if MODEL=='Measurement':
    start_time = time.time()
    # ===============Initialize============

    if os.path.exists(f'model_save/{Equation_name}/model_save_initial.pkl') == False:
        torch.save(Net.state_dict(),
                   f'model_save/{Equation_name}/model_save_initial.pkl')
        torch.save(optimizer.state_dict(),
                   f'model_save/{Equation_name}/optimizer_save_initial.pkl')
    if data_num != 0:
        if os.path.exists(f'model_save/{Equation_name}/model_save_pretrain_{data_num}_{noise_level}.pkl') == False:
            l_record = 1e8
            for iter in tqdm(range(pretrain_num)):
                optimizer.zero_grad()
                l_data = Loss.loss_data(DR_train, Y_train)
                l_data.backward()
                optimizer.step()
                if (iter + 1) % 1000 == 0:
                    if l_data.cpu().data.numpy() < l_record:
                        torch.save(Net.state_dict(),
                                   f'model_save/{Equation_name}/model_save_pretrain_{data_num}_{noise_level}.pkl')
                        torch.save(optimizer.state_dict(),
                                   f'model_save/{Equation_name}/optimizer_save_pretrain_{data_num}_{noise_level}.pkl')
                    else:
                        break
                    l_record = l_data.cpu().data.numpy()

    print('========Pretrain Finish!=========')

    measure_num=20000
    MSE_record=np.zeros([2,2,2])
    for bit_1 in [0, 1]:
        for bit_2 in [0, 1]:
            for bit_3 in [0, 1]:
                    l_record = 1e8
                    if data_num == 0:
                        Net.load_state_dict(
                            torch.load(f'model_save/{Equation_name}/model_save_initial.pkl'))
                        optimizer.load_state_dict(
                            torch.load(f'model_save/{Equation_name}/optimizer_save_initial.pkl'))
                    else:
                        Net.load_state_dict(
                            torch.load(f'model_save/{Equation_name}/model_save_pretrain_{data_num}_{noise_level}.pkl'))
                        optimizer.load_state_dict(
                            torch.load(
                                f'model_save/{Equation_name}/optimizer_save_pretrain_{data_num}_{noise_level}.pkl'))
                    for iter in tqdm(range(measure_num)):
                        optimizer.zero_grad()
                        reg_control = [bit_1, bit_2, bit_3]
                        if data_num==0:
                            if reg_control == [0, 0, 0]:
                                loss = Variable(torch.tensor([0.]), requires_grad=True)
                            else:
                                loss = 0
                        else:
                            loss=Loss.loss_data(DR_train,Y_train)

                        if reg_control[0] == 1:
                            l_inter = Loss.loss_inter(database_interior)
                            loss += lamda[0] * l_inter

                        if reg_control[1] == 1:
                            l_ini = Loss.loss_ini(database_ini, ref_ini)
                            loss += lamda[1] * l_ini

                        if reg_control[2] == 1:
                            l_b = Loss.loss_b(database_bl, database_br)
                            loss+= lamda[2] * l_b

                        loss.backward()
                        optimizer.step()
                        if (iter + 1) % 1000 == 0:
                            #prediction = Net(DR_valid).cpu().data.numpy()
                            # loss_valid = np.mean((Y_valid.cpu().data.numpy() - prediction) ** 2)
                            #print(loss.cpu().data.numpy())
                            if loss.cpu().data.numpy() > l_record:
                                break

                            l_record = loss.cpu().data.numpy()
                            torch.save(Net.state_dict(),
                                       f'model_save/{Equation_name}/model_save_{data_num}_{noise_level}.pkl')
                            torch.save(optimizer.state_dict(),
                                       f'model_save/{Equation_name}/optimizer_save_{data_num}_{noise_level}.pkl')

                    Net.load_state_dict(
                        torch.load(f'model_save/{Equation_name}/model_save_{data_num}_{noise_level}.pkl'))
                    prediction = Net(DR_valid).cpu().data.numpy()
                    loss_valid = np.mean((Y_valid.cpu().data.numpy() - prediction) ** 2)
                    MSE_record[bit_1, bit_2, bit_3] = loss_valid
                    print(f'reg:  {reg_control},  loss:   {loss.cpu().data.numpy()},  loss_valid:{loss_valid}')
                    time.sleep(0.1)

    np.save(f'result_save/{Equation_name}/MSE_data_{data_num}_{noise_level}.npy', MSE_record)
    end_time=time.time()
    with open("result_save/time.txt", "a") as f:
        f.write(f'{Equation_name}_{data_num}_{noise_level}:{end_time-start_time}s\n')
if MODEL=='SHAP':
    def plot_data_trend():
        All_shap=[]
        noise_level=0
        for data_num in [0,10,100,1000,10000]:
            result=np.load(f'result_save/{Equation_name}/MSE_data_{data_num}_{noise_level}.npy')
            print(result)
            print('--------------')
            shap_value=PINN_SHAP(result,plot_pic=False)
            All_shap.append(shap_value)
        All_shap=np.array(All_shap).T
        for i in range(All_shap.shape[0]):
            print(mk_test(All_shap[i])[0],mk_test(All_shap[i])[2])
        plt.figure(1, figsize=(2, 3), dpi=300)
        plt.style.use('ggplot')
        color_box = ['#2878b5', '#838AB4', '#f8ac8c', '#c82423', '#ff8884']
        x_plot=['$0$','$10^1$','$10^2$','$10^3$','$10^4$']
        reg_box = ['reg1', 'reg2', 'reg3', 'reg4']
        for i in range(3):
            plt.plot(All_shap[i], marker='x', color=color_box[i], label=reg_box[i])
            print(All_shap[i])
        plt.xticks(np.arange(0,5,1),x_plot,fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.ylim([-2.0,2.0])
        plt.legend(prop=font1, loc='lower left')
        # plt.savefig(f'fig_save/{Equation_name}/data_trend_reg_plot.tiff', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/{Equation_name}/data_trend_reg_plot.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig(f'fig_save/{Equation_name}/data_trend_reg_plot.pdf', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_noise_trend():
        All_shap=[]
        data_num=100
        for noise_level in [0,10,20,30,40,50]:
            result=np.load(f'result_save/{Equation_name}/MSE_data_{data_num}_{noise_level}.npy')
            shap_value=PINN_SHAP(result,plot_pic=False)
            All_shap.append(shap_value)
        All_shap=np.array(All_shap).T
        for i in range(All_shap.shape[0]):
            print(mk_test(All_shap[i])[0],mk_test(All_shap[i])[2])
        plt.figure(1, figsize=(2, 3), dpi=300)
        plt.style.use('ggplot')
        color_box = ['#2878b5', '#838AB4', '#f8ac8c', '#c82423', '#ff8884']
        x_plot=[0,0.1,0.2,0.3,0.4,0.5]
        reg_box = ['reg1', 'reg2', 'reg3', 'reg4']
        for i in range(3):
            plt.plot(All_shap[i], marker='x', color=color_box[i], label=reg_box[i])
            print(All_shap[i])
        plt.xticks(np.arange(0,6,1),x_plot,fontproperties='Arial', size=7)
        plt.yticks(fontproperties='Arial', size=7)
        plt.legend(prop=font1, loc='upper right')
        plt.ylim([-0.5, 1.5])
        plt.savefig(f'fig_save/{Equation_name}/noise_trend_reg_plot.tiff', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/{Equation_name}/noise_trend_reg_plot.jpg', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig_save/{Equation_name}/noise_trend_reg_plot.pdf', bbox_inches='tight', dpi=300)
        plt.show()
    plot_data_trend()
    plot_noise_trend()











