import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import set_seed
# from transformers import EsmTokenizer
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
# warnings.simplefilter()
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at facebook/esm2_t6_8M_UR50D were not used when initializing EsmForSequenceClassification")
set_seed(4)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 打印正在使用的设备

if device.type == 'cuda':
    print(torch.cuda.get_device_name(2))
model_checkpoint = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def AMP(file):
    test_sequences = file
    max_len = 60
    test_data = tokenizer(test_sequences, max_length=max_len, padding="max_length",truncation=True, return_tensors='pt')
    test_data = {key: value.to(device) for key, value in test_data.items()}
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

        def forward(self, x):
            with torch.no_grad():
                bert_output = self.bert(input_ids=x['input_ids'].to(device),
                                        attention_mask=x['attention_mask'].to(device))
            # 获取BERT模型的pooler输出
            output_feature = self.dropout(bert_output["logits"])
            output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
            output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
            output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
            output_feature = self.dropout(self.output_layer(output_feature))
            # return torch.sigmoid(output_feature),output_feature
            return output_feature

    model = MyModel()
    model.load_state_dict(torch.load("best_mic_model.pth", map_location=device),strict=False)
    model = model.to(device)
    # print("Model is on:", next(model.parameters()).device)
    # for key, value in test_data.items():
    #     print(f"{key} is on {value.device}")
    # print("Input data is on:", test_data.value.device)
    model.eval()
    out_probability = []
    with torch.no_grad():
        predict = model(test_data)
        # out_probability.extend(np.max(np.array(predict.cpu()),axis=1).tolist())
        # test_argmax = np.argmax(predict.cpu(), axis=1).tolist()
    # id2str = {0:"non-mic-AMP", 1:"mic-AMP"}
    return predict
def p1():
    B=[]
    seq=[]
    with open('seq.txt','r') as f:
        for line in f:
            # print(line)
            if len(line) < 5:
                continue
            str = line[0]
            for j in line:
                str += ' '
                str += j.upper()
            b = AMP(str)
            B.append(b)
            seq.append(line.strip())
    return B,seq

print(p1())