import torch
import torch.nn as nn
import os
import pickle
from time import sleep

class EEGNet_2s(nn.Module):
    def __init__(self, classes_num=3):
        super(EEGNet_2s, self).__init__()
        self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1, 64),bias=False),  # input shape (1, C, T)、output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(64,1),groups=8,bias=False),  # input shape (8, C, T)、output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1, 16),groups=16,bias=False),  # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1, 1),bias=False),  # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((240), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # 全连接
        x = x.contiguous().view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x
class EEGNet_3s(nn.Module):
    def __init__(self, classes_num=3):
        super(EEGNet_3s, self).__init__()
        self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1, 64),bias=False),  # input shape (1, C, T)、output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(64,1),groups=8,bias=False),  # input shape (8, C, T)、output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1, 16),groups=16,bias=False),  # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1, 1),bias=False),  # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((368), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # 全连接
        x = x.contiguous().view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x
class EEGNet_4s(nn.Module):
    def __init__(self, classes_num=3):
        super(EEGNet_4s, self).__init__()
        self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), bias=False),
            # input shape (1, C, T)、output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(64, 1), groups=8, bias=False),
            # input shape (8, C, T)、output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), groups=16, bias=False),
            # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False),
            # input shape (16, 1, T//4)、output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((496), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # 全连接
        x = x.contiguous().view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x
model_2s = EEGNet_2s()
model_3s = EEGNet_3s()
model_4s = EEGNet_4s()

class EEGNetMethod:
    def __init__(self):
        print("init EEGNetMethod")
        sleep(1)  # delay

    # def _band_filter(self, data, flow=0.5, fhigh=40, time_window=4):
    #     b, a = signal.butter(6, [2*flow/250, 2*fhigh/250], 'bandpass', analog=True)
    #     for iChan in range(data.shape[0]):
    #         data[iChan, :] = signal.filtfilt(b, a, data[iChan, :])
    #     return data

    def _predict(self, data, model):
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            output = model(data)
        _, predicted = torch.max(output, 1)
        return predicted.item()

    def recognize(self, data, personID=1):
        # # 只选取前59个eeg通道
        # data = data[:59]

        # 使用的时间窗(2s,3s,4s)
        time_window = int(data.shape[1] / 250)

        print("personID:", personID)

        # 加载该被试对应时间窗的模型
        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/eegnet_model/parameters_{}_{}.pkl'.format(personID, time_window)

        if time_window == 2:
            model = model_2s.double()
            # model_2s.load_state_dict(torch.load(path))
            # model_2s.double()
            # model = model_2s.double().eval()
        elif time_window == 3:
            model = model_3s.double()
            # model_3s.load_state_dict(torch.load(path))
            # model_3s.double()
            # model = model_3s.double().eval()
        elif time_window == 4:
            model = model_4s.double()
            # model_4s.load_state_dict(torch.load(path))
            # model_4s.double()
            # model = model_4s.double().eval()


        # with open(path, 'rb') as file:
        #     model = pickle.load(file)
        # model.load_state_dict(torch.load(path))
        # model.double()
        # model = eegnet.state_dict
        # model.eval()

        # # 滤波
        # pdata = self._band_filter(data, time_window=time_window)

        # 预测
        res = self._predict(data, model)
        res = int(res)
        return res