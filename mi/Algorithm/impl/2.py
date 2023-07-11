import pickle

import numpy as np

## 加载.kpl文件
#path = "./subject_data/sub1_block_3.pkl"
path = "E:\\PythonProject\\BCI_BC\\mi_debug_data\\测试数据\\S1\\block4.pkl"
# path = "./csp_model/parameters_1_2.pkl"
with open(path,"rb") as f:
    dataset = pickle.load(f)

data = dataset['data'][0:-1, :]
labels = dataset['data'][-1, :]
data = data
trigger_signal = labels
## 查找block的开始和结束索引
block_start_index = np.where(trigger_signal == 242)[0][0]
block_end_index = np.where(trigger_signal == 243)[0][0]
# 提取block数据
block_data = data[:,block_start_index:block_end_index]
block_data_labels = trigger_signal[block_start_index:block_end_index]

trial_start_index_11 = np.where(labels == 11)[0]
trial_start_index_12 = np.where(labels == 12)[0]
trial_start_index_13 = np.where(labels == 13)[0]
trial_start_index_21 = np.where(labels == 21)[0]
trial_start_index_22 = np.where(labels == 22)[0]
trial_start_index_23 = np.where(labels == 23)[0]
trial_start_index_31 = np.where(labels == 31)[0]
trial_start_index_32 = np.where(labels == 32)[0]
trial_start_index_33 = np.where(labels == 200)[0]
trial_start_index_241 = np.where(labels == 241)[0]
trial_start_index_2411 = np.where(labels == 241)[0]