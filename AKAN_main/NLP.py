from kan import KAN
from sklearn.model_selection import KFold,cross_val_score,train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import torch
import process as data
import numpy as np
from sklearn.metrics import accuracy_score

ac_p,label=data.deal()
aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
X=np.concatenate((aac,gaac,ac_p),axis=1)
print(X.shape)


y=label
y=np.array(y)
X = X.astype(np.float32)
y = y.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义并训练MLP模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.001, solver='adam', random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 进行交叉验证并计算每折的准确度
cross_val_accuracies = cross_val_score(mlp, X, y, cv=kf, scoring='accuracy')

# 打印每折的准确度
for fold, accuracy in enumerate(cross_val_accuracies, 1):
    print(f'第{fold}折的准确度: {accuracy:.4f}')

# 打印平均准确度
print(f'五折交叉验证的平均准确度: {cross_val_accuracies.mean():.4f}')