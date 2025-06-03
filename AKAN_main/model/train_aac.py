# 初始化一个列表来存储所有蛋白质的特征
import pandas as pd
from kan import KAN
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch

import numpy as np
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
protein_feature_train = []

# 打开并读取文件
with open('/tmp/pycharm_project_763/feature/feature_train/phychem_train_48.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        if line.startswith('>'):  # 跳过以'>'开头的行（序列标识行）
            continue
        # 分割行中的每个特征并转换为浮点数
        features = [float(x) for x in line.strip().split('\t')]
        # 将特征列表添加到主列表中
        protein_feature_train.append(features)

# 打印结果以验证
# print(protein_feature_list)

protein_feature_test = []

# 打开并读取文件
with open('/tmp/pycharm_project_763/feature/feature_test/phychem_test_48.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        if line.startswith('>'):  # 跳过以'>'开头的行（序列标识行）
            continue
        # 分割行中的每个特征并转换为浮点数
        features = [float(x) for x in line.strip().split('\t')]
        # 将特征列表添加到主列表中
        protein_feature_test.append(features)

df = pd.read_csv('/tmp/pycharm_project_763/data2/trainCPP.csv')

# 提取序列和标签
sequences = df['sequence'].tolist()
y = df['label'].tolist()

df2= pd.read_csv('/tmp/pycharm_project_763/data2/testCPP.csv')

# 提取序列和标签
sequences_test = df2['sequence'].tolist()
y_test = df2['label'].tolist()

# X = X.astype(np.float32)
# y = y.astype(np.float32)
# y_test = y_test.astype(np.float32)

X_train=np.array(protein_feature_train)
X_test=np.array(protein_feature_test)
y_train=np.array(y)
y_test=np.array(y_test)


X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)


y_train = y_train.reshape((y_train.shape[0], 1)).to(device)
y_test = y_test.reshape((y_test.shape[0], 1)).to(device)

dataset={}
dataset = {
        'train_input': X_train.to(device),
        'test_input': X_test.to(device),
        'train_label': y_train.to(device),
        'test_label': y_test.to(device)
    }
    # dataset =torch.from_numpy(dataset)

model = KAN(width=[48,5,1], grid=3, k=3,device=device)
model.to(device)
def train_acc():
    return torch.mean((torch.round(model(X_train)[:, 0]) == y_train[:, 0]).float())

def test_acc():
    return torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float())

results = model.train(dataset, opt="LBFGS", device=device,steps=10, metrics=(train_acc, test_acc))

print(results['train_acc'][-1], results['test_acc'][-1])