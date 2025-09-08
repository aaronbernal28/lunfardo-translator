import torch
import torch.nn as nn
import torch.nn.functional as F
from codebase import utils as ut
from transformers import BertModel

torch.cuda.empty_cache()

class model1(nn.Module):
    def __init__(self, name, embedding_dim=768, dropout=0.1):
        '''
        modelo naive basado en BERT + capas fully connected de referencia
        '''
        super().__init__()
        self.name = name
        self.pad_token = 0
        self.sep_token = 102
        self.BERT = BertModel.from_pretrained("bert-base-multilingual-uncased").eval()
        self.vocab_size = self.BERT.config.vocab_size
        self.embedding_dim = self.BERT.config.hidden_size
        self.linear1 = nn.Linear(self.embedding_dim, self.vocab_size//2)
        self.linear2 = nn.Linear(self.vocab_size//2, self.vocab_size)
        self.tanh = nn.Tanh()

    def step(self, input):
        '''
        input: (batch_size, seq_length) -- token ids
        output: (batch_size, vocab_size) -- logits
        '''
        with torch.inference_mode():
            bert_output = self.BERT(input).pooler_output # (batch_size, embedding_dim)

        output = self.linear1(bert_output)
        output = self.tanh(output)
        output = self.linear2(output)
        return output

    def forward(self, input):
        # truncado estrategico
        new_input_length = int(min(256, input.shape[1]))
        
        input = input[:, :new_input_length]
        target_pred_length = 512 - new_input_length
        batch_size = input.shape[0]

        target_pred = torch.zeros((batch_size, target_pred_length)).to(input.device)
        logits_pred = torch.zeros((batch_size, target_pred_length, self.vocab_size)).to(input.device)

        for i in range(1, target_pred_length):
            # deberian terminar en sep token y luego pad tokens.......x
            input_ids = torch.cat((input, target_pred[:, :i]), dim=1)
            logits_pred[:, i, :] = self.step(input_ids) # (batch_size, vocab_size)
            target_pred[:, i] = torch.argmax(logits_pred[:, i], dim=1) # (batch_size)

        return logits_pred, target_pred

    def loss(self, input, target):
        logits_pred, target_pred = self.forward(input)
        loss = torch.tensor(0, requires_grad=True).to(input.device)

        for i in range(1, min(target.shape[1], target_pred.shape[1])):
            loss += F.cross_entropy(logits_pred[:, i], target[:, i], ignore_index=self.pad_token)
            target_pred[:, i] = torch.argmax(logits_pred[:, i], dim=1)

        return loss
