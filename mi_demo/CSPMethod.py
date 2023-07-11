import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
import pickle
import mne
import warnings
from time import sleep

warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')


# 仅供测试所用


class CSPMethod:
    def __init__(self):
        print("init CSPMethod")
        sleep(1)  # delay
    
    def _band_fileter(self, data, flow=0.5, fhigh=40, time_window=4):
        # b, a = signal.butter(6, [2*flow/250, 2*fhigh/250], 'bandpass', analog=True)
        # for iChan in range(data.shape[0]):
        #     data[iChan, :] = signal.filtfilt(b, a, data[iChan, :])
        # return data
        
        info = mne.create_info(
            ch_names=["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
                      "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
                      "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
                      "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
                      "TP7", "TP8", 'Pz', "P3", "P4", "P5", "P6", "P7", "P8", "POz",
                      "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2"],  # "Pz", reference
            ch_types="eeg",  # channel type
            sfreq=250,  # frequency
            #
        )
        raw = mne.io.RawArray(data, info)  # create raw
        events = np.expand_dims(np.array([0, 0, 1]), axis=0)
        raw.info['events'].extend(events)
        event_id = {'0': 1}
        raw.filter(flow, fhigh, fir_design='firwin')
        train_epoches = mne.Epochs(raw, events, event_id, 0, time_window - 0.004,
                                   baseline=None, preload=True)
        fda = train_epoches.get_data()[0].astype(np.float32)
        return fda
    
    def _predict(self, data, fm1, fm2, fm3, lda1, lda2, lda3):
        fe1 = fm1.transform(data)
        fe2 = fm2.transform(data)
        fe3 = fm3.transform(data)
        val1 = lda1.predict(fe1)
        val2 = lda2.predict(fe2)
        val3 = lda3.predict(fe3)
        res = []
        
        for i in range(len(val1)):
            if val1[i] == 1 and val3[i] == 1:
                res.append(1)
            elif val1[i] == 1 and val3[i] == 2:
                res.append(3)
            elif val1[i] == 2 and val2[i] == 1:
                res.append(2)
            elif val1[i] == 2 and val2[i] == 2:
                res.append(3)
            else:
                res.append(7)
        res = np.array(res)
        return res
    
    def recognize(self, data, personID=1):
        # 只选取前59个eeg通道
        data = data[:59]
        # 使用的时间窗(2s,3s,4s)
        time_window = int(data.shape[1] / 250)

        print("personID:", personID)
        # 目前只有被试3的模型，都使用被试3的模型
        personID = 3
        # 加载该被试对应时间窗的模型
        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/csp_model/parameters_{}_{}.pkl'.format(personID, time_window)
        with open(path, 'rb') as f:
            parameters = pickle.load(f)
        # 解析参数
        freq_low, freq_high = parameters[0]
        fm1, fm2, fm3, lda1, lda2, lda3 = parameters[1]
        
        # 滤波
        pdata = self._band_fileter(data, freq_low, freq_high, time_window)
        
        # 预测
        res = self._predict(np.expand_dims(pdata, axis=0), fm1, fm2, fm3, lda1, lda2, lda3)
        res = np.int(res)
        return res
