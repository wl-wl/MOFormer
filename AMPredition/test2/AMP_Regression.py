import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')
set_seed(4)  
device = "cuda:0"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

# df_train = pd.read_csv('training_hemo_data.csv')
# df_val = pd.read_csv('val_hemo_data.csv')
df=pd.read_csv('mic_data.csv',usecols=['sequence','value'])
df_train,df_val=train_test_split(df,test_size=0.2,random_state=42)

train_sequences = df_train["sequence"].tolist()
train_labels = df_train["value"].tolist()
val_sequences = df_val["sequence"].tolist()
val_labels = df_val["value"].tolist()

class MyDataset(Dataset):
        def __init__(self,dict_data) -> None:
            super(MyDataset,self).__init__()
            self.data=dict_data
        def __getitem__(self, index):
            return [self.data['text'][index],self.data['labels'][index]]
        def __len__(self):
            return len(self.data['text'])

train_dict = {"text":train_sequences,'labels':train_labels}
val_dict = {"text":val_sequences,'labels':val_labels}


epochs = 5000
learning_rate = 0.00001
batch_size = 256

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def collate_fn(batch):
    max_len = 60
    pt_batch=tokenizer([b[0] for b in batch], max_length=max_len, padding="max_length",truncation=True, return_tensors='pt')
    labels=[b[1] for b in batch]
    return {'labels':labels,'input_ids':pt_batch['input_ids'],
            'attention_mask':pt_batch['attention_mask']}

train_data=MyDataset(train_dict)
val_data=MyDataset(val_dict)
train_dataloader=DataLoader(train_data,batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
val_dataloader=DataLoader(val_data,batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=1024)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0)

    def forward(self,x):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device))
          # 获取BERT模型的pooler输出
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        # return torch.sigmoid(output_feature),output_feature
        return output_feature

model = MyModel()
model = model.to(device)

# criterion = nn.BCELoss()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
train_epochs_acc = []
valid_epochs_acc = []

best_acc = 0
for epoch in range(epochs):
    model.train()
    train_epoch_loss = []
    currect = 0
    all_labels = []
    all_predictions = []
    for index, batch in enumerate(train_dataloader):
        batchs = {k: v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs= model(batchs)
        label = torch.tensor(batchs["labels"]).float()

        loss = criterion(outputs.to(device).view(-1), label.to(device).view(-1))
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())

        outputs_np = outputs.cpu().detach().numpy().flatten()
        labels_np = label.cpu().numpy().flatten()

        all_labels.extend(labels_np)
        all_predictions.extend(outputs_np)

        train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)



        for j in range(0,len(train_argmax)):
            if batchs["labels"][j]==train_argmax[j]:
                currect+=1
    train_acc = currect/len(train_labels)
    train_epochs_acc.append(train_acc)
    train_epochs_loss.append(np.average(train_epoch_loss))
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    print(f"Training RMSE: {mse}")
    print(f"Training RMSE: {rmse}")
    print(f"Training MAE: {mae}")
    print(f"Training R^2: {r2}")
    val_labels_list = []
    val_predictions_list = []
    model.eval()
    valid_epoch_loss = []
    with torch.no_grad():
        currect = 0
        for index, batch in enumerate(val_dataloader):
            batchs = {k: v for k, v in batch.items()}
            outputs = model(batchs)
            label = torch.tensor(batchs["labels"]).float()
            loss = criterion(outputs.to(device).view(-1), label.to(device).view(-1))
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())

            outputs_np = outputs.cpu().detach().numpy().flatten()
            labels_np = label.cpu().detach().numpy().flatten()
            val_labels_list.extend(labels_np)
            val_predictions_list.extend(outputs_np)

            val_argmax = np.argmax(outputs.cpu(), axis=1)
            for j in range(0,len(val_argmax)):
                if batchs["labels"][j]==val_argmax[j]:
                    currect+=1

    valid_epochs_loss.append(np.average(valid_epoch_loss))
    val_acc = currect/len(val_labels)
    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),f"best_model.pth")
    valid_epochs_acc.append(val_acc)
    mse = mean_squared_error(val_labels_list, val_predictions_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(val_labels_list, val_predictions_list)
    r2 = r2_score(val_labels_list, val_predictions_list)

    print(f"test MSE: {mse}")
    print(f"test RMSE: {rmse}")
    print(f"test MAE: {mae}")
    print(f"test R^2: {r2}")
    # print('',train_epochs_loss,valid_epochs_loss)
    print(f'epoch:{epoch}, train_epochs_loss:{np.average(train_epochs_loss)}, valid_epochs_loss:{np.average(valid_epochs_loss)}')

length_1=len(outputs)

y_pred = outputs.view(-1).cpu().numpy()
y_true = label.view(-1).cpu().numpy()# 计算误差
# error = [abs(y_pred[i]-y_true[i])for i in range(len(y_pred))] # 绘制散点图
# plt.scatter(y_pred, y_true, c=error, cmap='coolwarm')
# plt.xlabel('Predicted Values')
# plt.ylabel('True Values')
# plt.colorbar()
# plt.show()

x=list(np.arange(length_1))
plt.plot(x, y_pred, lw=2, ls='-', c='b')
plt.plot(x, y_true, lw=2, ls='-', c='r')
plt.show()



