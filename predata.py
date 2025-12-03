import numpy as np
import pandas as pd
from Utils import std, Eu_dis, construct_H_with_KNN_from_distance
from sklearn import model_selection
import torch


def read_data():
    ori_data = pd.read_excel(r"./data/7.balanced data.xlsx").values
    zeros = np.zeros((ori_data.shape[0], 1))
    print(zeros.shape)
    ori_data = np.c_[ori_data, zeros]
    ori_data[273:341, 12] = 1
    ori_data[537:586, 12] = 1
    np.random.shuffle(ori_data)
    x = std(ori_data[:, :11])
    y = ori_data[:, 11:12]  # 586:0(341)//1(245)   训练集：验证集 8：2  117 68//49
    dis_mat = Eu_dis(x)
    H = construct_H_with_KNN_from_distance(dis_mat, k_neig=10)
    return torch.tensor(x, dtype=torch.float32),  torch.tensor(y, dtype=torch.float32),  H, torch.tensor(ori_data[:, 12:], dtype=torch.float32)














