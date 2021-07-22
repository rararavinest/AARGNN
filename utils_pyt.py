import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


#addres



#get weather data
def getweather():
    hourlyweather='/home/sw/smalldata/hourlyweather.txt'
    data0=np.loadtxt(hourlyweather)
    #repeat as hour to 5min
    # data0=torch.FloatTensor(data)
    data12=np.concatenate([data0,data0,data0,data0,data0,data0,
                      data0,data0,data0,data0,data0,data0],axis=1)
    data_weather=data12.reshape(25632,-1)
    print("weather data0:", data0.shape)
    print("weather:", data_weather.shape)
    return data_weather


'''def getnodeattr():
    trafficaddr='/home/sw/smalldata/Traffic_all.txt'
    occupancyaddr='/home/sw/smalldata/Occu_all.txt'
    traffic=np.loadtxt(trafficaddr)
    occupancy=np.loadtxt(occupancyaddr, delimiter=' ')
    f_traffic=torch.FloatTensor(traffic)
    f_occu=torch.FloatTensor(occupancy)
    node_attr=torch.cat([f_traffic, f_occu], dim=0)
    return node_attr'''

def getnodeattr():
    trafficaddr = '/home/sw/smalldata/Traffic_all.txt'
    traffic=np.loadtxt(trafficaddr)
    # f_traffic=torch.FloatTensor(traffic)
    node_attr=traffic
    return node_attr

def getpoivector():
    poiaddr = '/home/sw/smalldata/poidensity.txt'
    poi_vec=np.loadtxt(poiaddr, delimiter=',')
    poimatrix=np.ones([30,30])
    for i in range(30):
        for j in range(i+1,30):
            d=poi_vec[:,i]-poi_vec[:,j]
            poimatrix[i,j]=np.sqrt(np.dot(d,d))
    return poimatrix


def getedgeattr():
    adjaddr='/home/sw/smalldata/W_Matrix.txt'
    inciaddr='/home/sw/smalldata/inci_all.txt'
    # poiaddr='/home/sw/smalldata/poidensity.txt'
    adj=np.loadtxt(adjaddr, delimiter=',')
    inci=np.loadtxt(inciaddr)
    poi=getpoivector()
    a=adj
    a=a.reshape(1,900,1)
    i=inci.reshape(-1,900,1)
    print("inci shape:", i.shape)
    p=poi
    p=p.reshape(1,900,1)
    print("poi shape:",p.shape)
    times=inci.shape[0]
    print("times:",times)
    aa=a.repeat(times,0)
    pp=p.repeat(times,0)
    edge_attr=np.concatenate([i,aa,pp],axis=2)
    print("edge shape:",edge_attr.shape)
    return edge_attr

def getglobalattr():
    # hourlyweather = '/home/sw/smalldata/hourlyweather.txt'
    weather=getweather()
    # f_w=torch.FloatTensor(weather)
    return weather


def getgraphattr(batch_size):
    datat=getnodeattr()
    dataw=getedgeattr()
    datai=getglobalattr()

    tlength = datat.shape[0]
    wlength = dataw.shape[0]
    print("wlength:",wlength)
    ilength = datai.shape[0]
    # judge t w i
    ydata = np.zeros_like(datat)
    ydata[:-1] = datat[1:]
    ydata[-1] = datat[0]
    # ydata=torch.FloatTensor(ydata)

    datalength = datat.shape[0]
    index_t = datalength / 10 * 7 - 1
    index_t = math.floor(index_t)
    # 35682
    index_v = datalength / 10 * 9 - 1
    index_v = math.floor(index_v)
    # 45877
    traindata_size = index_t + 1
    testdata_size = index_v - index_t
    validdata_size = datalength - 1 - index_v

    ttraindata = datat[0:index_t].copy()
    wtraindata = dataw[0:index_t].copy()
    itraindata = datai[0:index_t].copy()
    ytraindata = ydata[0:index_t].copy()

    ttestdata = datat[index_t + 1:index_v].copy()
    wtestdata = dataw[index_t + 1:index_v].copy()
    itestdata = datai[index_t + 1:index_v].copy()
    ytestdata = ydata[index_t + 1:index_v].copy()

    tvaliddata = datat[index_v + 1:].copy()
    wvaliddata = dataw[index_v + 1:].copy()
    ivaliddata = datai[index_v + 1:].copy()
    yvaliddata = ydata[index_v + 1:].copy()

    # tensor format
    tfeat_traindata = torch.FloatTensor(ttraindata)
    wfeat_traindata = torch.FloatTensor(wtraindata)
    ifeat_traindata = torch.FloatTensor(itraindata)
    yfeat_train = torch.FloatTensor(ytraindata)

    tfeat_testdata = torch.FloatTensor(ttestdata)
    wfeat_testdata = torch.FloatTensor(wtestdata)
    ifeat_testdata = torch.FloatTensor(itestdata)
    yfeat_test = torch.FloatTensor(ytestdata)

    tfeat_validdata = torch.FloatTensor(tvaliddata)
    wfeat_validdata = torch.FloatTensor(wvaliddata)
    ifeat_validdata = torch.FloatTensor(ivaliddata)
    yfeat_valid = torch.FloatTensor(yvaliddata)

    # dataset
    train_data = TensorDataset(tfeat_traindata, wfeat_traindata, ifeat_traindata, yfeat_train)
    test_data = TensorDataset(tfeat_testdata, wfeat_testdata, ifeat_testdata, yfeat_test)
    print("t:",tfeat_validdata.shape)
    print("w:",wfeat_validdata.shape)
    print("p:",ifeat_validdata.shape)
    print("target:",yfeat_valid.shape)
    valid_data = TensorDataset(tfeat_validdata, wfeat_validdata, ifeat_validdata, yfeat_valid)

    # dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # return featin
    return train_data_loader, valid_data_loader, test_data_loader

def definedloss(predict, target, loss_type):
    if(loss_type=='mse'):
        crierion = torch.nn.MSELoss()
        mseloss=crierion(predict, target)
        loss=torch.sqrt(mseloss)
    elif(loss_type=='mae'):
        crierion=torch.nn.L1Loss()
        maeloss=crierion(predict, target)
        loss=maeloss
    elif(loss_type=='mape'):
        mask=target!=0
        loss=np.fabs((target[mask]-predict[mask])/target[mask]).mean()
    else:
        crierion = torch.nn.MSELoss()
        mseloss = crierion(predict, target)
        loss = torch.sqrt(mseloss)
    return loss









