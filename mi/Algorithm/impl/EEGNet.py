import os
import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import pickle


"""预处理和清洗数据"""
# 数据降采样：从1000hz到250hz
def downsample(data,original_freq,target_freq):
    # 计算降采样的比例
    downsample_ratio = original_freq // target_freq

    # 计算新的采样点数量
    new_length = data.shape[1] // downsample_ratio

    # 初始化降采样后的数据数组
    downsampled_data = np.zeros((data.shape[0],new_length))

    # 对每个通道进行降采样
    for i in range(data.shape[0]):
        # 使用scipy的decimate函数进行降采样
        downsampled_data[i] = signal.decimate(data[i],downsample_ratio)

    return downsampled_data

# 样本和标签的提取函数
def preprocess_data_and_labels(data,labels):
    ## 待裁剪的数据和标签
    data = data
    trigger_signal = labels
    ## 查找block的开始和结束索引
    block_start_index = np.where(trigger_signal == 242)[0][0]
    block_end_index = np.where(trigger_signal == 243)[0][0]
    # 提取block数据
    block_data = data[:, block_start_index:block_end_index]
    block_data_labels = trigger_signal[block_start_index:block_end_index]  # 对应的标签

    ## 所有标签开始索引
    start_index_11 = np.where(block_data_labels == 11)[0]
    start_index_12 = np.where(block_data_labels == 12)[0]
    start_index_13 = np.where(block_data_labels == 13)[0]
    start_index_21 = np.where(block_data_labels == 21)[0]
    start_index_22 = np.where(block_data_labels == 22)[0]
    start_index_23 = np.where(block_data_labels == 23)[0]
    start_index_31 = np.where(block_data_labels == 31)[0]
    start_index_32 = np.where(block_data_labels == 32)[0]
    start_index_33 = np.where(block_data_labels == 33)[0]
    ## 结束标签索引
    end_index = np.where(block_data_labels == 241)[0]

    trial_start_index_11_21_31 = np.concatenate([start_index_11, start_index_21, start_index_31])
    trial_start_indexs = np.sort(trial_start_index_11_21_31)

    # 提取30个trial的数据和标签
    ## 获取每个trial结束的索引
    trial_end_indexs = []
    for i in range(2, len(end_index), 3):
        trial_end_index = end_index[i]  # 获取第i个结束标签
        trial_end_indexs.append(trial_end_index)
    ## 提取每个trial的数据和标签
    trial_data = []
    trial_data_labels = []
    for trial_start, trial_end in zip(trial_start_indexs, trial_end_indexs):
        trial_data_i = block_data[:, trial_start:trial_end + 1]
        trial_data_labels_i = block_data_labels[trial_start:trial_end + 1]

        trial_data.append(trial_data_i)
        trial_data_labels.append(trial_data_labels_i)

    # 分别提取每个trial中前2s、前3s、前4s的运动想象数据和标签
    trial_data_2s = []
    trial_data_2s_labels = []
    trial_data_3s = []
    trial_data_3s_labels = []
    trial_data_4s = []
    trial_data_4s_labels = []


    # 遍历每个trial
    for i in range(len(trial_data)):
        trial_data_i = trial_data[i]
        trial_data_labels_i = trial_data_labels[i]

        # 查找截止标签为241的索引
        end_index_241 = np.where(trial_data_labels_i == 241)[0]

        # 提取前2s的数据和标签
        trial_data_2s.append(trial_data_i[:, :end_index_241[0] + 1])
        trial_data_2s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])

        # 提取前3s的数据和标签
        trial_data_3s.append(trial_data_i[:, :end_index_241[1] + 1])
        trial_data_3s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])


        # 提取前4s的数据和标签
        trial_data_4s.append(trial_data_i[:, :end_index_241[2] + 1])
        trial_data_4s_labels.append(trial_data_labels_i[:end_index_241[0] + 1])


    # 标签转换（11,12,12---0：左手， 21,22,23---1：右手， 31,32,33---2：脚）
    ## 定义转换规则的映射字典
    label_map = {11: "0", 12: "0", 13: "0", 21: "1", 22: "1", 23: "1", 31: "2", 32: "2", 33: "2"}
    ## 转换列表中的每个数组为标签
    # 前2s数据的标签
    labels_2s = []
    for array in trial_data_2s_labels:
        label = label_map[array[0]]
        labels_2s.append(label)
    # 前3s数据的标签
    labels_3s = []
    for array in trial_data_2s_labels:
        label = label_map[array[0]]
        labels_3s.append(label)
    # 前4s数据的标签
    labels_4s = []
    for array in trial_data_2s_labels:
        label = label_map[array[0]]
        labels_4s.append(label)

    return trial_data_2s,labels_2s


# 定义一个空列表，用于加载单个受试者的3个不同block数据
loaded_data = []

## 遍历block.pkl文件
directory = "E:\\PythonProject\\BCI_BC\\mi_debug_data\\训练数据\\S1"
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        n = filename[5]
        file_path = os.path.join(directory,filename)

        ###加载未处理的block文件
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)

        ### 从加载的block文件中获取数据和标签
        data = dataset['data'][0:-1,:]
        labels = dataset['data'][-1,:]
        ### 对获取的数据和标签进行裁剪
        data, label = preprocess_data_and_labels(data, labels)
        ### 将数据和标签组合在一起
        data_and_labels = {'data': data, 'labels': label}

        #保存为.pkl文件
        output_file = 'subject_data/sub1_block_{}.pkl'.format(n)
        with open(output_file, 'wb') as file:
            pickle.dump(data_and_labels, file)

        ### 将加载的数据和标签添加到列表中
        loaded_data.append(data_and_labels)

# 拼接三个列表的数据和标签
## 数据
list1 = loaded_data[0]['data']
list2 = loaded_data[1]['data']
list3 = loaded_data[2]['data']
## 标签
list4 = loaded_data[0]['labels']
list5 = loaded_data[1]['labels']
list6 = loaded_data[2]['labels']

# 创建空列表存储拼接后的数据和标签
merged_data_list = []
merged_labels_list = []

## 遍历每个列表将其元素添加到合并列表中
### 添加数据
for array in list1:
    merged_data_list.append(array)
for array in list2:
    merged_data_list.append(array)
for array in list3:
    merged_data_list.append(array)
### 添加标签
for array in list4:
    merged_labels_list.append(array)
for array in list5:
    merged_labels_list.append(array)
for array in list6:
    merged_labels_list.append(array)

# 将合并后的数据存为.pkl文件
## 将合并后的数据和标签组合在一起
merged_data_and_labels = {'data':merged_data_list, 'labels':merged_labels_list}
## 保存为.pkl文件
output_file = 'subject_data/sub1_block_all.pkl'
with open(output_file,'wb') as f:
    pickle.dump(merged_data_and_labels,f)


# 最终作为数据集的数据和标签
data = merged_data_list   #（列表形式）
labels = merged_labels_list
encoded_labels = [int(label) for label in labels]   # 将字符标签转换为整数


data = np.array(data)  #（数组形式）
label = np.array(encoded_labels)


"""自定义数据集类"""
##继承Dataset类
class EEGDataset(Dataset):
    def __init__(self,data,labels):
        self.data = torch.tensor(data,dtype=torch.float32)
        self.labels = torch.tensor(labels,dtype=torch.long)

    def __getitem__(self,index):
        sample = self.data[index]
        label = self.labels[index]
        return sample,label

    def __len__(self):
        return len(self.data)

# 实例化自定义数据集对象
eegdataset = EEGDataset(data,label)

"""加载数据集"""
dataloader = DataLoader(eegdataset,batch_size=9,shuffle=True)

"""搭建神经网络"""
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

        self.out = nn.Linear((1024), classes_num)

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

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(eegnet.parameters(),lr=learning_rate)
# # 设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# eegnet.to(device)

"""训练模型"""
##设置一些参数
total_train_step = 0
num_epochs = 10

for epoch in range(num_epochs):
    print("--------第 {} 轮训练开始--------".format(epoch+1))

    ## 训练步骤开始
    for data,label in dataloader:
        data = data.unsqueeze(1)   # .to(device)
        labels = label  # .to(device)

        outputs = eegnet(data)
        loss = loss_fn(outputs,labels)

        ### 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数：{}，loss:{}".format(total_train_step,loss.item()))

# 保存模型参数
# torch.save(eegnet,"./eegnet_model/model_3_4.pkl")
# model = torch.load("./eegnet_model/model_3_4.pkl")
# print(model)

torch.save(eegnet.state_dict(),"./eegnet_model/parameters_1_2.pkl")
eegnet.load_state_dict(torch.load('./eegnet_model/parameters_1_2.pkl'))
print(eegnet.state_dict)