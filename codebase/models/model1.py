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
        self.BERT = BertModel.from_pretrained("bert-base-multilingual-uncased").eval()
        self.vocab_size = self.BERT.config.vocab_size
        self.embedding_dim = self.BERT.config.hidden_size
        self.linear1 = nn.Linear(self.embedding_dim, self.vocab_size//2)
        self.linear2 = nn.Linear(self.vocab_size//2, self.vocab_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        '''
        input: (batch_size, seq_length) -- token ids
        output: (batch_size, vocab_size) -- logits
        '''
        with torch.inference_mode():
            bert_output = self.BERT(input).pooler_output # (batch_size, embedding_dim)

        res = self.linear1(bert_output)
        res = self.tanh(res)
        res = self.linear2(res)
        return res

    def loss(self, input, target):
        # truncado estrategico
        input_proportional_length = input.shape[1] / (input.shape[1] + target.shape[1])
        new_input_length = int(512 * input_proportional_length)
        
        input = input[:, :new_input_length]
        target = target[:, :(512 - new_input_length)]

        return None
    