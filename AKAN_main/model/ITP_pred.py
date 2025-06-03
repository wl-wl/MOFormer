from kan2 import KAN
from sklearn.model_selection import KFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import process as data
import numpy as np

ac_p,label=data.deal()
aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
X=np.concatenate((aac,gaac,ac_p),axis=1)
print(X.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = KAN(width=[37,3,1], grid=3, k=3)
fold = 1

y=label
y=np.array(y)
X = X.astype(np.float32)
y = y.astype(np.float32)

num_cross_val=1

dataset={}
for fold in range(num_cross_val):
    X_train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
    X_test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
    y_train = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
    y_test = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])



    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train).view(-1, 1)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test).view(-1, 1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    dataset = {
        'train_input': X_train,
        'test_input': X_test,
        'train_label': y_train,
        'test_label': y_test
    }
    # dataset =torch.from_numpy(dataset)

    def train_acc():
        return torch.mean((torch.round(model(X_train)[:, 0]) == y_train).float())

    def test_acc():
        return torch.mean((torch.round(model(X_test)[:, 0]) == y_test).float())

    # results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc))
    results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc))

    print(results['train_acc'][-1], results['test_acc'][-1])


    fold += 1


