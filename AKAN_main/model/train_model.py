import os
import pandas as pd
import moviepy
import moviepy.video.io.ImageSequenceClip
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
import random
import umap
from scipy.interpolate import interp1d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score,roc_curve
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import csv
from kan import KAN
batch_size = 32  # Setting batchsize
learning_rates=0.000065 # Setting learning rates
d_m="1024_480"#Selecting the dimension mapping method: 480_1024, 480_1024

device = torch.device("cuda:1")
print(torch.cuda.get_device_name(1))



def compute_metrics(predictions, labels):
    """
    计算常用的二分类评估指标。
    :param predictions: 模型的预测输出。
    :param labels: 真实标签。
    :return: 一个包含准确率、精确度、召回率和F1分数的字典。
    """
    # 转换预测和标签为一维数组
    predictions = predictions[:, 0]  # 假设预测结果是二维的，每行一个预测
    labels = labels[:, 0]

    # 计算准确率
    accuracy = torch.mean((torch.round(predictions) == labels).float()).item()

    # 将Tensor转换为NumPy数组用于sklearn的计算
    predictions_np = torch.round(predictions).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算精确度、召回率和F1分数
    precision = precision_score(labels_np, predictions_np)
    recall = recall_score(labels_np, predictions_np)
    f1 = f1_score(labels_np, predictions_np)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}



class MyDataset(Dataset):
    def __init__(self, file):
        self.sequence, self.label = self.read_file(file)
        self.sequence_protbert = self.add_space_between_characters(self.sequence)

    def read_file(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            data = list(csv_reader)
            random.seed(42)
            random.shuffle(data)
            for row in data:
                sequences.append(row[0])
                labels.append(row[1])
        return sequences, labels

    def add_space_between_characters(self, input_list):
        new_list = []
        for element in input_list:
            new_element = ' '.join(element)
            new_list.append(new_element)
        return new_list

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        sample = self.sequence[index]
        sample_protbert = self.sequence_protbert[index]
        label = int(self.label[index])
        return sample, label, sample_protbert


train_file = '/tmp/pycharm_project_763/data2/trainCPP.csv'
train_dataset = MyDataset(train_file)
X,y,X_p=train_dataset.sequence,train_dataset.label,train_dataset.sequence_protbert
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


test_file = '/tmp/pycharm_project_763/data2/testCPP.csv'
test_dataset = MyDataset(test_file)
X_val,y_val,X_p_val=test_dataset.sequence,test_dataset.label,test_dataset.sequence_protbert

def etract(X,y,X_p):
    device = torch.device("cuda:1")
    model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    tokenizer_pro = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model_pro = BertModel.from_pretrained("Rostlab/prot_bert")
    # model_pro.to(device)
    dropout = nn.Dropout(0.2)
    fc_pro = nn.Linear(1024, 480)

    inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    encoded_input = tokenizer_pro(X_p, padding=True, truncation=True,return_tensors='pt')
    outputs_pro = model_pro(**encoded_input)
    pooler_output1 = outputs.pooler_output
    pooler_output2=outputs_pro.pooler_output
    pooler_output2=fc_pro(pooler_output2)
    pooler_output2=pooler_output2.to(device)
    # X =pooler_output1+pooler_output2
    X=pooler_output1
    # pooler_output1 = pooler_output1.cpu().detach().numpy()
    # pooler_output2 = pooler_output2.cpu().detach().numpy()
    #
    # # 使用PCA降至100维
    # pca = PCA(n_components=100)
    # reduced_pooler_output1 = pca.fit_transform(pooler_output1)
    # reduced_pooler_output2 = pca.fit_transform(pooler_output2)
    #
    # # 将numpy数组转换回张量，并传输回设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reduced_pooler_output1 = torch.tensor(reduced_pooler_output1).to(device)
    # reduced_pooler_output2 = torch.tensor(reduced_pooler_output2).to(device)
    #
    # # 将降维后的张量相加
    # X = reduced_pooler_output1 + reduced_pooler_output2
    #
    X = X.cpu().detach().numpy()
    y = np.array(y)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X,y



#
# pooler_output1=pooler_output1.cpu().detach().numpy()
# pooler_output2=pooler_output2.cpu().detach().numpy()
# tsne = TSNE(n_components=20,method='exact')
# reduced_pooler_output1 = tsne.fit_transform(pooler_output1)
# reduced_pooler_output2 = tsne.fit_transform(pooler_output2)
#
# # 将numpy数组转换回张量，并传输回设备
# reduced_pooler_output1 = torch.tensor(reduced_pooler_output1).to(device)
# reduced_pooler_output2 = torch.tensor(reduced_pooler_output2).to(device)
#
# # 将降维后的张量相加
# X = reduced_pooler_output1 + reduced_pooler_output2



# ID=1
# df = pd.read_csv('/tmp/pycharm_project_763/data2/trainCPP.csv')
# DSSP=[]
# # 提取序列和标签
# sequences = df['sequence'].tolist()
# y = df['label'].tolist()
# for seq in sequences:
#     dssp=np.load("/tmp/pycharm_project_763/feature/feature_train/dssp/" +str(ID)+".npy")
#     dssp=np.sum(dssp, axis=0)
#     DSSP.append(dssp)
#     ID+=1
# DSSP= torch.tensor(DSSP).to(device)
#
# X=torch.cat((X, DSSP), dim=1)


X_train,y_train=etract(X,y,X_p)
X_test,y_test=etract(X_val,y_val,X_p_val)


# X_train=np.array(X_train)
# X_test=np.array(X_train)
# y_train=np.array(X_train)
# y_test=np.array(X_train)
#480
model = KAN(width=[480,5,1], grid=3, k=2,device=device)
fold = 1



image_folder="/tmp/pycharm_project_763/results/"
dataset={}


X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

y_train = y_train.reshape((y_train.shape[0], 1)).to(device)
y_test = y_test.reshape((y_test.shape[0], 1)).to(device)

dataset = {
    'train_input': X_train.to(device),
    'test_input': X_test.to(device),
    'train_label': y_train.to(device),
    'test_label': y_test.to(device)
}
# dataset =torch.from_numpy(dataset)

model.to(device)
# print(model)

def train_acc():
    return torch.mean((torch.round(model(X_train)[:, 0]) == y_train[:, 0]).float())

def test_acc():
    return torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float())


def compute_metrics():
    """
    计算常用的二分类评估指标。
    :param predictions: 模型的预测输出。
    :param labels: 真实标签。
    :return: 一个包含准确率、精确度、召回率和F1分数的字典。
    """
    # 转换预测和标签为一维数组
    predictions = torch.round(model(X_test)[:, 0])  # 假设预测结果是二维的，每行一个预测
    labels = y_test[:, 0].float()

    # 计算准确率
    accuracy = torch.mean((torch.round(predictions) == labels).float()).item()

    # 将Tensor转换为NumPy数组用于sklearn的计算
    predictions_np = torch.round(predictions).cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    # 计算精确度、召回率和F1分数
    precision = precision_score(labels_np, predictions_np)
    recall = recall_score(labels_np, predictions_np)
    f1 = f1_score(labels_np, predictions_np)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# results = model.train(dataset, opt="LBFGS", device=device,steps=10, metrics=(train_acc, test_acc),
#                       save_fig=True, img_folder=image_folder,in_vars= list(range(1, 10)), out_vars=['CPP'])
results = model.train(dataset, opt="LBFGS", device=device, steps=10, metrics=(train_acc, test_acc))
# print("------",torch.mean((torch.round(model(X_depen)[:, 0]) == y_depen[:, 0]).float()))
# results = model.train(dataset, opt="LBFGS",
#     steps=100,
#     log=10,
#     lamb=1e-4,
#     lamb_l1=1e-4,
#     lamb_entropy=1.,
#     lr=1e-3,
#     batch=32,
#     device=device, metrics=(train_acc, test_acc))
# model.save_ckpt(name=f'epoch_{fold + 1}.pth')

video_name = 'video'
fps = 10
fps = fps
files = os.listdir(image_folder)
train_index = []
for file in files:
    if file[0].isdigit() and file.endswith('.jpg'):
        train_index.append(int(file[:-4]))
train_index = np.sort(train_index)
image_files = [image_folder + '/' + str(train_index[index]) + '.jpg' for index in train_index]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_gif(video_name + '.gif')

print(results['train_acc'][-1], results['test_acc'][-1])



# model.plot(beta=100, scale=1)
# video_name='video'
# fps=10
# fps = fps
# image_folder="image"
# files = os.listdir(image_folder)
print(model)
# model.save_ckpt("KAN","/tmp/pycharm_project_763/result")
# model=KAN.load_ckpt("KAN","/tmp/pycharm_project_763/result")
# print(model)
model.plot(beta=100, scale=1)
#解决过拟合问题
model = model.prune()





# import os
# import pandas as pd
# import moviepy
# import moviepy.video.io.ImageSequenceClip
# 
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from transformers import BertModel, BertTokenizer
# import random
# import umap
# from scipy.interpolate import interp1d
# import numpy as np
# import torch
# from torch.utils.data2 import Dataset, DataLoader
# from sklearn.metrics import roc_auc_score,roc_curve
# import torch
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import torch.nn as nn
# from torch.utils.data2 import Dataset, DataLoader
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import random
# import matplotlib.pyplot as plt
# from torch.utils.data2.sampler import SubsetRandomSampler
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score
# import torch
# import csv
# from kan import KAN
# batch_size = 32  # Setting batchsize
# learning_rates=0.000065 # Setting learning rates
# d_m="1024_480"#Selecting the dimension mapping method: 480_1024, 480_1024
# 
# device = torch.device("cuda:1")
# print(torch.cuda.get_device_name(1))
# 
# 
# 
# def compute_metrics(predictions, labels):
#     """
#     计算常用的二分类评估指标。
#     :param predictions: 模型的预测输出。
#     :param labels: 真实标签。
#     :return: 一个包含准确率、精确度、召回率和F1分数的字典。
#     """
#     # 转换预测和标签为一维数组
#     predictions = predictions[:, 0]  # 假设预测结果是二维的，每行一个预测
#     labels = labels[:, 0]
# 
#     # 计算准确率
#     accuracy = torch.mean((torch.round(predictions) == labels).float()).item()
# 
#     # 将Tensor转换为NumPy数组用于sklearn的计算
#     predictions_np = torch.round(predictions).cpu().numpy()
#     labels_np = labels.cpu().numpy()
# 
#     # 计算精确度、召回率和F1分数
#     precision = precision_score(labels_np, predictions_np)
#     recall = recall_score(labels_np, predictions_np)
#     f1 = f1_score(labels_np, predictions_np)
# 
#     return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
# 
# 
# 
# class MyDataset(Dataset):
#     def __init__(self, file):
#         self.sequence, self.label = self.read_file(file)
#         self.sequence_protbert = self.add_space_between_characters(self.sequence)
# 
#     def read_file(self, file_path):
#         sequences = []
#         labels = []
#         with open(file_path, 'r', newline='') as csv_file:
#             csv_reader = csv.reader(csv_file)
#             next(csv_reader, None)
#             data2 = list(csv_reader)
#             random.seed(42)
#             random.shuffle(data2)
#             for row in data2:
#                 sequences.append(row[0])
#                 labels.append(row[1])
#         return sequences, labels
# 
#     def add_space_between_characters(self, input_list):
#         new_list = []
#         for element in input_list:
#             new_element = ' '.join(element)
#             new_list.append(new_element)
#         return new_list
# 
#     def __len__(self):
#         return len(self.sequence)
# 
#     def __getitem__(self, index):
#         sample = self.sequence[index]
#         sample_protbert = self.sequence_protbert[index]
#         label = int(self.label[index])
#         return sample, label, sample_protbert
# 
# 
# train_file = '/tmp/pycharm_project_763/data2/trainCPP.csv'
# train_dataset = MyDataset(train_file)
# X,y,X_p=train_dataset.sequence,train_dataset.label,train_dataset.sequence_protbert
# 
# test_file = '/tmp/pycharm_project_763/data2/testCPP.csv'
# test_dataset = MyDataset(train_file)
# X_val,y_val,X_p_val=test_dataset.sequence,test_dataset.label,test_dataset.sequence_protbert
# 
# def etract(X):
#     device = torch.device("cuda:1")
#     model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
#     model.to(device)
#     tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
#     tokenizer_pro = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
#     model_pro = BertModel.from_pretrained("Rostlab/prot_bert")
#     # model_pro.to(device)
#     dropout = nn.Dropout(0.2)
#     fc_pro = nn.Linear(1024, 480)
# 
#     inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt")
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     encoded_input = tokenizer_pro(X_p, padding=True, truncation=True,return_tensors='pt')
#     outputs_pro = model_pro(**encoded_input)
#     pooler_output1 = outputs.pooler_output
#     pooler_output2=outputs_pro.pooler_output
#     pooler_output2=fc_pro(pooler_output2)
#     pooler_output2=pooler_output2.to(device)
#     X =pooler_output1+pooler_output2
# 
#     pooler_output1 = pooler_output1.cpu().detach().numpy()
#     pooler_output2 = pooler_output2.cpu().detach().numpy()
# 
#     # 使用PCA降至100维
#     pca = PCA(n_components=100)
#     reduced_pooler_output1 = pca.fit_transform(pooler_output1)
#     reduced_pooler_output2 = pca.fit_transform(pooler_output2)
# 
#     # 将numpy数组转换回张量，并传输回设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     reduced_pooler_output1 = torch.tensor(reduced_pooler_output1).to(device)
#     reduced_pooler_output2 = torch.tensor(reduced_pooler_output2).to(device)
# 
#     # 将降维后的张量相加
#     X = reduced_pooler_output1 + reduced_pooler_output2
#     return X
# 
# 
# 
# #
# # pooler_output1=pooler_output1.cpu().detach().numpy()
# # pooler_output2=pooler_output2.cpu().detach().numpy()
# # tsne = TSNE(n_components=20,method='exact')
# # reduced_pooler_output1 = tsne.fit_transform(pooler_output1)
# # reduced_pooler_output2 = tsne.fit_transform(pooler_output2)
# #
# # # 将numpy数组转换回张量，并传输回设备
# # reduced_pooler_output1 = torch.tensor(reduced_pooler_output1).to(device)
# # reduced_pooler_output2 = torch.tensor(reduced_pooler_output2).to(device)
# #
# # # 将降维后的张量相加
# # X = reduced_pooler_output1 + reduced_pooler_output2
# 
# 
# 
# # ID=1
# # df = pd.read_csv('/tmp/pycharm_project_763/data2/trainCPP.csv')
# # DSSP=[]
# # # 提取序列和标签
# # sequences = df['sequence'].tolist()
# # y = df['label'].tolist()
# # for seq in sequences:
# #     dssp=np.load("/tmp/pycharm_project_763/feature/feature_train/dssp/" +str(ID)+".npy")
# #     dssp=np.sum(dssp, axis=0)
# #     DSSP.append(dssp)
# #     ID+=1
# # DSSP= torch.tensor(DSSP).to(device)
# #
# # X=torch.cat((X, DSSP), dim=1)
# 
# 
# X=etract(X)
# # X_depen=etract(X_depen)
# kf = KFold(n_splits=5, shuffle=True, random_state=100)
# #480
# model = KAN(width=[100,7,1], grid=3, k=3,device=device)
# fold = 1
# 
# 
# X = X.cpu().detach().numpy()
# y = np.array(y)
# 
# X = X.astype(np.float32)
# y = y.astype(np.float32)
# 
# num_cross_val=5
# image_folder="/tmp/pycharm_project_763/results/"
# dataset={}
# for fold in range(num_cross_val):
#     X_train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
#     X_test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
#     y_train = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
#     y_test = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
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
#     # print(model)
# 
#     def train_acc():
#         return torch.mean((torch.round(model(X_train)[:, 0]) == y_train[:, 0]).float())
# 
#     def test_acc():
#         return torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float())
# 
# 
#     def compute_metrics():
#         """
#         计算常用的二分类评估指标。
#         :param predictions: 模型的预测输出。
#         :param labels: 真实标签。
#         :return: 一个包含准确率、精确度、召回率和F1分数的字典。
#         """
#         # 转换预测和标签为一维数组
#         predictions = torch.round(model(X_test)[:, 0])  # 假设预测结果是二维的，每行一个预测
#         labels = y_test[:, 0].float()
# 
#         # 计算准确率
#         accuracy = torch.mean((torch.round(predictions) == labels).float()).item()
# 
#         # 将Tensor转换为NumPy数组用于sklearn的计算
#         predictions_np = torch.round(predictions).cpu().detach().numpy()
#         labels_np = labels.cpu().detach().numpy()
# 
#         # 计算精确度、召回率和F1分数
#         precision = precision_score(labels_np, predictions_np)
#         recall = recall_score(labels_np, predictions_np)
#         f1 = f1_score(labels_np, predictions_np)
# 
#         return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
# 
#     results = model.train(dataset, opt="LBFGS", device=device,steps=10, metrics=(train_acc, test_acc),
#                           save_fig=True, img_folder=image_folder,in_vars= list(range(1, 10)), out_vars=['CPP'])
#     # results = model.train(dataset, opt="LBFGS", device=device, steps=10, metrics=(train_acc, test_acc))
#     # print("------",torch.mean((torch.round(model(X_depen)[:, 0]) == y_depen[:, 0]).float()))
# 
#     # model.save_ckpt(name=f'epoch_{fold + 1}.pth')
# 
#     video_name = 'video'
#     fps = 10
#     fps = fps
#     files = os.listdir(image_folder)
#     train_index = []
#     for file in files:
#         if file[0].isdigit() and file.endswith('.jpg'):
#             train_index.append(int(file[:-4]))
#     train_index = np.sort(train_index)
#     image_files = [image_folder + '/' + str(train_index[index]) + '.jpg' for index in train_index]
#     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
#     clip.write_gif(video_name + '.gif')
# 
#     print(results['train_acc'][-1], results['test_acc'][-1])
#     # print(compute_metrics())
# 
#     fold += 1
# 
#     model.plot(beta=100, scale=1)
# # video_name='video'
# # fps=10
# # fps = fps
# # image_folder="image"
# # files = os.listdir(image_folder)
# print(model)
# # model.save_ckpt("KAN","/tmp/pycharm_project_763/result")
# # model=KAN.load_ckpt("KAN","/tmp/pycharm_project_763/result")
# # print(model)
# model.plot(beta=100, scale=1)
# #解决过拟合问题
# model = model.prune()
# print(torch.mean((torch.round(model(X_test)[:, 0]) == y_test[:, 0]).float()))