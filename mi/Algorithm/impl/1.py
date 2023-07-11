import pickle
import torch
import numpy as np
from torch import nn


# ## 加载.kpl文件
# #path = "./subject_data/sub1_block_3.pkl"
# #path = "C:\\Users\\lilei\\Desktop\\BCI_BC\\mi_debug_data\\训练数据\\S1\\block1.pkl"
# # path = "./csp_model/parameters_1_2.pkl"
# path = "./eegnet_model/model_1_2.pkl"
# with open(path,"rb") as f:
#     dataset = pickle.load(f)
#
# a = 1
# b = 2
class EEGNet(nn.Module):
    def __init__(self, classes_num=3):
        super(EEGNet, self).__init__()
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

        self.out = nn.Linear((2016), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # 全连接
        x = x.contiguous().view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x

eegnet = EEGNet()

eegnet = torch.load("./eegnet_model/model_3_4.pkl")

A = 1





# data = dataset['data'][0:-1, :]
# labels = dataset['data'][-1, :]
#
# ## 待裁剪的数据和标签
# data = data
# trigger_signal = labels
# ## 查找block的开始和结束索引
# block_start_index = np.where(trigger_signal == 242)[0][0]
# block_end_index = np.where(trigger_signal == 243)[0][0]
# # 提取block数据
# block_data = data[:,block_start_index:block_end_index]
# block_data_labels = trigger_signal[block_start_index:block_end_index]  #对应的标签
#
# ## 所有标签开始索引
# start_index_11 = np.where(block_data_labels == 11)[0]
# start_index_12 = np.where(block_data_labels == 12)[0]
# start_index_13= np.where(block_data_labels == 13)[0]
# start_index_21= np.where(block_data_labels == 21)[0]
# start_index_22 = np.where(block_data_labels == 22)[0]
# start_index_23 = np.where(block_data_labels == 23)[0]
# start_index_31= np.where(block_data_labels == 31)[0]
# start_index_32 = np.where(block_data_labels == 32)[0]
# start_index_33 = np.where(block_data_labels == 33)[0]
# ## 结束标签索引
# end_index = np.where(block_data_labels == 241)[0]
#
# trial_start_index_11_21_31 = np.concatenate([start_index_11, start_index_21, start_index_31])
# trial_start_indexs = np.sort(trial_start_index_11_21_31)
#
# # 提取30个trial的数据和标签
# ## 获取每个trial结束的索引
# trial_end_indexs = []
# for i in range(2,len(end_index),3):
#     trial_end_index = end_index[i]  # 获取第i个结束标签
#     trial_end_indexs.append(trial_end_index)
# ## 提取每个trial的数据和标签
# trial_data = []
# trial_data_labels = []
# for trial_start,trial_end in zip(trial_start_indexs,trial_end_indexs):
#     trial_data_i = block_data[:,trial_start:trial_end+1]
#     trial_data_labels_i = block_data_labels[trial_start:trial_end+1]
#
#     trial_data.append(trial_data_i)
#     trial_data_labels.append(trial_data_labels_i)
#
# # 分别提取每个trial中前2s、前3s、前4s的运动想象数据和标签
# trial_data_2s = []
# trial_data_2s_labels = []
# trial_data_3s = []
# trial_data_3s_labels = []
# trial_data_4s = []
# trial_data_4s_labels = []
#
# # 遍历每个trial
# for i in range(len(trial_data)):
#     trial_data_i = trial_data[i]
#     trial_data_labels_i = trial_data_labels[i]
#
#     # 查找截止标签为241的索引
#     end_index_241 = np.where(trial_data_labels_i == 241)[0]
#
#     # 提取前2s的数据和标签
#     trial_data_2s.append(trial_data_i[:, :end_index_241[0] + 1])
#     trial_data_2s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])
#
#     # 提取前3s的数据和标签
#     trial_data_3s.append(trial_data_i[:, :end_index_241[1] + 1])
#     trial_data_3s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])
#
#     # 提取前4s的数据和标签
#     trial_data_4s.append(trial_data_i[:, :end_index_241[2] + 1])
#     trial_data_4s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])
#
# # 标签转换（11,12,12---0：左手， 21,22,23---1：右手， 31,32,33---2：脚）
# ## 定义转换规则的映射字典
# label_map = {11: "0", 12: "0", 13: "0", 21: "1", 22: "1", 23: "1", 31: "2", 32: "2", 33: "2"}
# ## 转换列表中的每个数组为标签
# # 前2s数据的标签
# labels_2s = []
# for array in trial_data_2s_labels:
#     label = label_map[array[0]]
#     labels_2s.append(label)
# # 前3s数据的标签
# labels_3s = []
# for array in trial_data_2s_labels:
#     label = label_map[array[0]]
#     labels_3s.append(label)
# # 前4s数据的标签
# labels_4s = []
# for array in trial_data_2s_labels:
#     label = label_map[array[0]]
#     labels_4s.append(label)



