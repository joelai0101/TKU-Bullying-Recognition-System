import torch
import numpy as np
from transformers import BertTokenizer, BertModel, logging
import pandas as pd
from torch import nn
logging.set_verbosity_warning()

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def predict(text):

    model_dict_path = './offensive language classification/models/model_bert.pth'

    model_dict = torch.load(model_dict_path)

    model = BertClassifier()
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    unlabeled_input = tokenizer(
        text, padding='max_length', max_length=32, truncation=True, return_tensors="pt")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    unlabeled_mask = unlabeled_input['attention_mask'].to(device)
    unlabeled_input_id = unlabeled_input['input_ids'].squeeze(1).to(device)

    unlabeled_output = model(unlabeled_input_id, unlabeled_mask)
    pseudo_label = unlabeled_output.argmax(dim=1).item()
    # print(text, pseudo_label)

    if pseudo_label:
        label = 'Non-offensive'
    else: 
        label = 'Offensive'

    return unlabeled_output, label
