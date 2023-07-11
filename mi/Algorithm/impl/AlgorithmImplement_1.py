from Algorithm.AlgorithmInterface import AlgorithmInterface
from scipy import signal
import numpy as np
import math
from Algorithm.impl.EEGNetMethod import EEGNetMethod

class AlgorithmImplement(AlgorithmInterface):
    def __init__(self):
        super().__init__()

        # trial开始trigger，题目说明中给出
        self.trial_start_trig = 200
        
        # 定义采样率，题目文件中给出
        samp_rate = 250
        # 计算偏移时间（s）
        offset_time = 0.00
        # 偏移长度
        self.offset_len = math.floor(offset_time * samp_rate)
        # 计算长度
        self.cal_len_2 = 2 * samp_rate
        self.cal_len_3 = 3 * samp_rate
        self.cal_len_4 = 4 * samp_rate
        # 每个trial 计算次数[1,2,3]
        self._trial_cal_time = 0
        
        # # 预处理滤波器设置
        # self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        
        # 初始化算法
        self.method = EEGNetMethod()

    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        while not end_flag:
            data_model = self.comm_proxy.get_data()
            if data_model is None:
                continue
            data_model.data = np.array(data_model.data)
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)
            else:
                # 计算模式，则进行处理
                cal_flag, result = self.__cal_proc(data_model)
                # 如果有结果，则进行报告
                if result is not None:
                    self.comm_proxy.report(result)
                    # 清空缓存
                    # self.__clear_cache()
            end_flag = data_model.finish_flag
        print('Algo Run End')
    
    """事件检测"""
    def __idle_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导（第65导是标签）
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_start_trig)[0]
        # 脑电数据
        eeg_data = data[0: -1, :]
        if len(trigger_idx) > 0:
            # 有trial开始trigger则进行计算
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos: eeg_data.shape[1]]
        else:
            # 没有trial开始trigger则不进行计算
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    """进行计算，并报告结果"""
    def __cal_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        personID = data_model.subject_id
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_start_trig)[0]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 如果trigger为空，表示依然在当前试次中，根据数据长度判断是否计算
        # print(self.cache_data.shape[1])
        if len(trigger_idx) == 0:
            # 当已缓存的数据大于等于所需要使用的计算数据时，进行计算
            if self.cache_data.shape[1] >= self.cal_len_4 and self._trial_cal_time == 2:
                self._trial_cal_time = 0
                # 获取所需计算长度的数据
                use_data = self.cache_data[:, self.offset_len: int(self.cal_len_4)]
                # # 滤波处理
                # use_data = self.__preprocess(use_data)
                # 开始计算，返回计算结果
                result = self.method.recognize(use_data,personID)
                print("4s结果",result)
                # 停止计算模式
                cal_flag = False
            elif self.cache_data.shape[1] >= self.cal_len_3 and self._trial_cal_time == 1:
                self._trial_cal_time += 1
                # 获取所需计算长度的数据
                use_data = self.cache_data[:, self.offset_len: int(self.cal_len_3)]
                # # 滤波处理
                # use_data = self.__preprocess(use_data)
                # 开始计算，返回计算结果
                result = self.method.recognize(use_data,personID)
                print("3s结果",result)
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                cal_flag = True
            elif self.cache_data.shape[1] >= self.cal_len_2 and self._trial_cal_time == 0:
                self._trial_cal_time += 1
                # 获取所需计算长度的数据
                use_data = self.cache_data[:, self.offset_len: int(self.cal_len_2)]
                # # 滤波处理
                # use_data = self.__preprocess(use_data)
                # 开始计算，返回计算结果
                result = self.method.recognize(use_data,personID)
                print("2s结果",result)
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                cal_flag = True
            else:
                # 反之继续采集数据
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                result = None
                cal_flag = True
        # 下一试次已经开始,需要强制结束计算
        else:
            # 下一个trial开始trigger的位置
            next_trial_start_trig_pos = trigger_idx[0]
            # 如果拼接该数据包中部分的数据后，可以满足所需要的计算长度，则拼接数据达到所需要的计算长度
            # 如果拼接完该trial的所有数据后仍无法满足所需要的数据长度，则只能使用该trial的全部数据进行计算
            use_len = min(next_trial_start_trig_pos, self.cal_len - self.cache_data.shape[1])
            self.cache_data = np.append(self.cache_data, eeg_data[:, 0: use_len], axis=1)
            # 考虑偏移量
            use_data = self.cache_data[:, self.offset_len: self.cache_data.shape[1]]
            # # 滤波处理
            # use_data = self.__preprocess(use_data)
            # 开始计算
            result = self.method.recognize(use_data)
            # 开始新试次的计算模式
            cal_flag = True
            # 清除缓存的数据
            self.__clear_cache()
            # 添加新试次数据
            self.cache_data = eeg_data[:, next_trial_start_trig_pos: eeg_data.shape[1]]
        return cal_flag, result

    # def __get_pre_filter(self, samp_rate):
    #     fs = samp_rate
    #     f0 = 50             #滤波器的截止频率
    #     q = 35              #滤波器的质量因子
    #     b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    #     return b, a

    def __clear_cache(self):
        self.cache_data = None

    # def __preprocess(self, data):
    #     # # 选择使用的导联
    #     # data = data[self.select_channel, :]
    #     filter_data = signal.filtfilt(self.filterB, self.filterA, data)
    #     return filter_data
