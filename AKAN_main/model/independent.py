
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

import numpy as np
import torch
import csv
from kan import KAN
device = torch.device("cuda:2")
print(torch.cuda.get_device_name(2))



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
X =pooler_output1+pooler_output2

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

model2 = KAN(width=[480,7,1], grid=3, k=3,device=device)

print(model2)
model2.load_ckpt("epoch_2.pth")

print(X.shape)

X = X.cpu().detach().numpy()
y = np.array(y)

X = X.astype(np.float32)
y = y.astype(np.float32)




X_train = torch.from_numpy(X).to(device)
y_train = torch.from_numpy(y).to(device)


y_train = y_train.reshape((y_train.shape[0], 1)).to(device)

model2.to(device)
print(model2)

dataset = {
    'train_input': X_train.to(device),
    'train_label': y_train.to(device),
    'test_input': X_train.to(device),
    'test_label': y_train.to(device),

}


def train_acc():
    return torch.mean((torch.round(model2(X_train)[:, 0]) == y_train[:, 0]).float())




model2.train(dataset, opt="LBFGS", device=device, steps=0, save_fig=True)

with torch.no_grad():
    predictions=torch.round(model2(X_train)[:, 0])

print(torch.mean((predictions == y_train[:, 0]).float()))