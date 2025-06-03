import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from kan import KAN
from sklearn.model_selection import KFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from sklearn.preprocessing import StandardScaler
import process as data
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# y = y.reshape((y.shape[0], 1))
# X = [' '.join(map(str, x)) for x in X]  # 将特征转换为字符串形式


# scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 使用TensorDataset将特征和标签组合
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# 初始化tokenizer和model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#
#
# # 定义一个简单的分类模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits


model = BertClassifier(bert_model)


# 数据编码函数
def encode_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_accuracies = []
test_accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = encode_data(X_train, y_train)
    test_dataset = encode_data(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    model.train()
    for epoch in range(3):  # 训练3个epoch
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 计算训练集准确度
    model.eval()
    train_preds = []
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_accuracies.append(train_accuracy)

    # 计算测试集准确度
    test_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_accuracies.append(test_accuracy)

    print(f'Fold train accuracy: {train_accuracy:.4f}, test accuracy: {test_accuracy:.4f}')

print(f'平均训练准确度: {np.mean(train_accuracies):.4f}')
print(f'平均测试准确度: {np.mean(test_accuracies):.4f}')
