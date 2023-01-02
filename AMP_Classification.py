import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
set_seed(4)  
device = "cuda:0"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"

df_train = pd.read_csv('./training_data.csv')
df_val = pd.read_csv('./val_data.csv')

train_sequences = df_train["Seq"].tolist()
train_labels = df_train["Label"].tolist()
val_sequences = df_val["Seq"].tolist()
val_labels = df_val["Label"].tolist()

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


epochs = 100
learning_rate = 0.00001
batch_size = 256

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def collate_fn(batch):
    max_len = 30
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
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=320)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.output_layer = nn.Linear(64,2)
        self.dropout = nn.Dropout(0)

    def forward(self,x):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device)) 
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        # return torch.sigmoid(output_feature),output_feature
        return torch.softmax(output_feature,dim=1),output_feature


model = MyModel()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
    for index, batch in enumerate(train_dataloader):
        batchs = {k: v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs, _ = model(batchs)
        label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64), num_classes=2).float()
        loss = criterion(outputs.to(device), label.to(device))
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        for j in range(0,len(train_argmax)):
            if batchs["labels"][j]==train_argmax[j]:
                currect+=1
    train_acc = currect/len(train_labels)
    train_epochs_acc.append(train_acc)
    train_epochs_loss.append(np.average(train_epoch_loss))
    
    model.eval()
    valid_epoch_loss = []
    with torch.no_grad():
        currect = 0
        for index, batch in enumerate(val_dataloader):
            batchs = {k: v for k, v in batch.items()}
            outputs,output_feature = model(batchs)
            label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64), num_classes=2).float()
            loss = criterion(outputs.to(device), label.to(device))
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
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

    print(f'epoch:{epoch}, train_acc:{train_acc}, val_acc:{val_acc}')
