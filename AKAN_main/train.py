from kan import KAN
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import process as data
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ac_p,label=data.deal()
aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
X=np.concatenate((aac,gaac,ac_p),axis=1)
print(X.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)



model = KAN(width=[37,5,1], grid=3, k=3)
fold = 1

y=label
y=np.array(y)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
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


model.to(device)
def train_acc():
    return torch.mean((torch.round(model(X_train)[:, 0]) == y_train[:, 0]).float())

def test_acc():
    return torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float())

results = model.train(dataset, opt="LBFGS", device=device,steps=10, metrics=(train_acc, test_acc))

print(results['train_acc'][-1], results['test_acc'][-1])

num_cross_val=5
# for fold in range(num_cross_val):
#     X_train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
#     X_test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
#     y_train = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
#     y_test = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
#
#     X_train = torch.from_numpy(X_train).to(device)
#     y_train = torch.from_numpy(y_train).to(device)
#     X_test = torch.from_numpy(X_test).to(device)
#     y_test = torch.from_numpy(y_test).to(device)
#
#     y_train = y_train.reshape((y_train.shape[0], 1)).to(device)
#     y_test = y_test.reshape((y_test.shape[0], 1)).to(device)
#
#     dataset = {
#         'train_input': X_train.to(device),
#         'test_input': X_test.to(device),
#         'train_label': y_train.to(device),
#         'test_label': y_test.to(device)
#     }
#     # dataset =torch.from_numpy(dataset)
#
#     model.to(device)
#     def train_acc():
#         return torch.mean((torch.round(model(X_train)[:, 0]) == y_train[:, 0]).float())
#
#     def test_acc():
#         return torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float())
#
#     results = model.train(dataset, opt="LBFGS", device=device,steps=20, metrics=(train_acc, test_acc))
#
#     print(results['train_acc'][-1], results['test_acc'][-1])
#
#
#     fold += 1
#
# image_folder = 'video_img'
# model = KAN(width=[37,1], grid=3, k=3)
# model = KAN(width=[4, 5, 3], grid=5, k=3, seed=0, device=device)
# model(dataset['train_input'])
# model.plot(beta=100, scale=1, in_vars=['SL', 'SW', 'PL', 'PW'], out_vars=['Set', 'Ver', 'Vir'])


# model = model.prune()
# model(dataset['X_train'])
# model.plot()