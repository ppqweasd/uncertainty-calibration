import pandas as pd
import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.signal import square


from collections import deque

#importing the dataset
mu=0
sigma=10
x1 = np.arange(0, 20 * np.pi, 0.1)
#y = np.sin(x) + np.cos(x) * x
#scaling and selecting data
#close_prices=np.loadtxt('I:\\dalta2.csv',delimiter=',')
close_prices=np.sin(x1)#+random.gauss(mu,sigma)#np.loadtxt('I:\\YNQ_ac',delimiter=',')[:12345]

scaler = StandardScaler()

close_prices_arr = np.array(close_prices).reshape(-1, 1)
close_prices = scaler.fit_transform(close_prices_arr)



window_size=20

def create_timestamps_ds(series,timestep_size=window_size):
    time_stamps = []
    labels = []
    aux_deque = deque(maxlen=timestep_size)

    #  starting the timestep deque
    for i in range(timestep_size):
        aux_deque.append(0)

        #feed the timestamps list
    for i in range(len(series)-1):
        aux_deque.append(series[i])
        time_stamps.append(list(aux_deque))

        #feed the labels lsit
    for i in range(len(series)-1):
        labels.append(series[i + 1])

    assert len(time_stamps) == len(labels), "Something went wrong"

    #torch-tensoring it
    features = torch.tensor(time_stamps[timestep_size:]).float()
    labels = torch.tensor(labels[timestep_size:]).float()

    return features, labels


@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm_1 = BayesianLSTM(1, 100)
        self.linear = nn.Linear(100, 1)
        self.lstm_3 = nn.LSTM(1, 100)
        self.lstm_4 = nn.LSTM(100, 100)

    def forward(self, x):

        x_, _ = self.lstm_1(x)
        x_, _ = self.lstm_4(x_)

        x_, _ = self.lstm_4(x_)
        #gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_

Xs, ys = create_timestamps_ds(close_prices)
#X_train=Xs
#y_train=ys
X_train, X_test, y_train, y_test = train_test_split(Xs,ys,test_size=.25,random_state=42,shuffle=False)#, y_train, y_test = train_test_split(Xs,ys,test_size=.25,random_state=42,shuffle=False)

#
# y_train=ys


ds = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True)

net=NN()

'''
pretrained_dict = torch.load('156.pkl')
model_dict=net.state_dict()#'linear.weight','linear.bias','lstm_4.weight_ih_l0','lstm_4.weight_hh_l0','lstm_4.bias_ih_l0','lstm_4.bias_hh_l0']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
net.load_state_dict(pretrained_dict )
'''

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def custom_update(param, grad, lr):
    # 自定义更新逻辑
    param.data -= lr * grad



iteration = 0
for epoch in range(10):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = net.sample_elbo(inputs=datapoints,
                               labels=labels,
                               criterion=criterion,
                               sample_nbr=30)
        loss.backward()
        # for name, param in net.named_parameters():
        #     if "lstm_1.weight_ih_rho" in name:  # 替换为你的参数名称
        #         custom_update(param, param.grad, lr=0.01)
        #     else:
        #         # 其他参数按照常规方式进行更新
        #         pass
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            preds_test = net(X_test)[:, 0].unsqueeze(1)
            loss_test = criterion(preds_test, y_test)
            print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))

for name, param in net.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

preds_test = net(X_test)[:, 0].unsqueeze(1)
loss_test = criterion(preds_test, y_test)
print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
#torch.save(net.state_dict(),'156.pkl')


'''
for name,value in net.named_parameters():

    print(name,value)


    lstm_1.weight_ih_mu
lstm_1.weight_ih_rho
lstm_1.weight_hh_mu
lstm_1.weight_hh_rho
lstm_1.bias_mu
lstm_1.bias_rho





df_pred = pd.DataFrame(original)
df_pred["Date"] = df.Date
df["Date"] = pd.to_datetime(df_pred["Date"])
df_pred = df_pred.reset_index()
'''


def pred_stock_future(X_test,future_length,sample_nbr=10):

#sorry for that, window_size is a global variable, and so are X_train and Xs
    global window_size
    global X_train
    global Xs
    global scaler

    #creating auxiliar variables for future prediction
    preds_test = []
    test_begin = X_test[0:1, :, :]
    test_deque = deque(test_begin[0,:,0].tolist(), maxlen=window_size)

    #idx_pred = np.arange(len(X_train), len(Xs))
    idx_pred = np.arange(len(X_test))
    #print(len(Xs))
    #print(len(idx_pred))
    #predict it and append to list
    for i in range(len(X_test)):
    #print(i)
        as_net_input = torch.tensor(test_deque).unsqueeze(0).unsqueeze(2)
        pred = [net(as_net_input).cpu().item() for i in range(sample_nbr)]


        test_deque.append(torch.tensor(pred).mean().cpu().item())
        preds_test.append(pred)

        if i % future_length == 0:
        #our inptus become the i index of our X_test
        #That tweak just helps us with shape issues
            test_begin = X_test[i:i+1, :, :]
            test_deque = deque(test_begin[0,:,0].tolist(), maxlen=window_size)

            #preds_test = np.array(preds_test).reshape(-1, 1)
            #preds_test_unscaled = scaler.inverse_transform(preds_test)

    return idx_pred, preds_test
def get_confidence_intervals(preds_test, ci_multiplier):
    global scaler

    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)

    k=scaler.inverse_transform(y_test.detach().cpu().numpy().reshape(-1, 1))
    #print(k)
    m=pred_mean.cpu().numpy().reshape(-1, 1)
    #print(m)
    err=np.abs(k-m)

    #print(err)




    pred_std = preds_test.std(1).detach().cpu().numpy()

    #print(pred_std)
    #corr=np.corrcoef(err,pred_std)
    correlation_coefficient, p_value = pearsonr(err.flatten(),pred_std.flatten())

    print('*********')
    print(correlation_coefficient, p_value)

    pred_std = torch.tensor((pred_std))

    upper_bound = pred_mean + (pred_std * ci_multiplier)
    lower_bound = pred_mean - (pred_std * ci_multiplier)
    #gather unscaled confidence intervals

    pred_mean_final = pred_mean.unsqueeze(1).detach().cpu().numpy()
    pred_mean_unscaled = scaler.inverse_transform(pred_mean_final)

    upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
    upper_bound_unscaled = scaler.inverse_transform(upper_bound_unscaled)

    lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
    lower_bound_unscaled = scaler.inverse_transform(lower_bound_unscaled)

    return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled


future_length=10
sample_nbr=30
ci_multiplier=20

names = locals()

for i in range(1):
    names['noisy' + str(i)] = (np.random.rand(np.shape(X_test)[0],np.shape(X_test)[1],np.shape(X_test)[2])-0.5)  * 3 * i+np.array(X_test).astype(
        'float64')
    #error, MC_error, MC_pred, MC_std, ll = best_network.predict(names['noisy' + str(i)], y_test)


    idx_pred, preds_test = pred_stock_future(names['noisy' + str(i)], future_length, sample_nbr)
    T = 10000


    ll = (logsumexp(-0.5  * (y_test.numpy() - np.array(preds_test)) ** 2., 0) - np.log(T) - 0.5 * np.log(2 * np.pi) )
    test_ll = np.mean(ll)

    print(test_ll)

pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled = get_confidence_intervals(preds_test,ci_multiplier)
#y = np.array(close_prices[-750:]).reshape(-1, 1)
#under_upper = upper_bound_unscaled > y
#over_lower = lower_bound_unscaled < y
#total = (under_upper == over_lower)

#print("{} our predictions are in our confidence interval".format(np.mean(total)))
#     params = {"ytick.color" : "black","xtick.color" : "black","axes.labelcolor" : "black","axes.edgecolor" : "black"}
#     plt.rcParams.update(params)
#     plt.xlabel('working time(s)')
#     plt.ylabel('pressure(kPa)')
# #plt.title("IBM Stock prices", color="white")
#
#     plt.plot(np.arange(len(original)),original,color='black',label="Real")
#
#     plt.plot(idx_pred,pred_mean_unscaled,label="Prediction for {} minute, than consult".format(future_length),color="red")
#
#     plt.fill_between(x=idx_pred,y1=upper_bound_unscaled[:,0],y2=lower_bound_unscaled[:,0],facecolor='green',label="Confidence interval",alpha=0.5)
#
#
#
#     plt.legend()
#     plt.savefig('I:\\LSTM_ori_complete'+str(i)+'.jpg',dpi=1000)
#     plt.show()

params = {"ytick.color" : "black","xtick.color" : "black","axes.labelcolor" : "black","axes.edgecolor" : "black"}
plt.rcParams.update(params)

#plt.title("IBM Stock prices", color="white")
#plt.xlabel('working time(min)')
#plt.ylabel('pressure(kPa)')
#plt.ylim((-20, 80))
plt.fill_between(x=idx_pred,y1=upper_bound_unscaled[:,0],y2=lower_bound_unscaled[:,0],facecolor='green',label="Confidence interval",alpha=0.75)

plt.plot(idx_pred,scaler.inverse_transform(y_test.detach().cpu().numpy()[-len(pred_mean_unscaled[:,0]):]),label="Real",alpha=1,color='black',linewidth=0.5)

plt.plot(idx_pred,pred_mean_unscaled,label="Prediction",color="red",alpha=0.5)

plt.legend(loc='upper right')
plt.savefig('J:\\SIN_cal4.jpg',dpi=1000)
plt.show()