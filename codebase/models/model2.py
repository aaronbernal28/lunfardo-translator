import torch
import torch.nn as nn
import torch.nn.functional as F
from codebase import utils as ut
from transformers import BertModel

torch.cuda.empty_cache()

class model2(nn.Module):
    def __init__(self, name):
        '''
        modelo basado en BERT + transformer
        '''
        super().__init__()
        self.name = name
        self.pad_token = torch.tensor(0)
        self.sep_token = torch.tensor(102)
        self.cls_token = torch.tensor(101)

        # cargar embeddings de BERT
        self.embeddings = BertModel.from_pretrained("bert-base-multilingual-uncased").eval().embeddings.word_embeddings.weight.detach()
        torch.cuda.empty_cache()

        self.vocab_size = self.embeddings.shape[0]
        self.emb_dim = self.embeddings.shape[1]

        self.transformer = nn.Transformer(d_model=self.emb_dim, batch_first=True)
        self.linear = nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, input, target):
        '''
        input: (batch_size, input_seq_length)
        target: (batch_size, target_seq_length)

        input_emb: (batch_size, input_seq_length, emb_dim)
        target_emb: (batch_size, target_seq_length, emb_dim)

        scores: (batch_size, target_seq_length, vocab_size)
        '''
        input_emb = self.embeddings[input]
        target_emb = self.embeddings[target]
        
        # target_mask evita que el transformer vea futuros tokens
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_emb.size(1)).to(target_emb.device)

        scores = self.transformer.forward(src=input_emb, tgt=target_emb,
                                          tgt_mask=target_mask,
                                          src_key_padding_mask=(input == self.pad_token),
                                          tgt_key_padding_mask=(target == self.pad_token)) # (batch_size, target_seq_length, emb_dim)
        scores = self.linear(scores)
        return scores
    
    def generate(self, input, max_length=512):
        '''
        input: (input_seq_length)
        '''
        input = input.unsqueeze(0) # (1, input_seq_length)
        input_emb = self.embeddings[input]
        target_pred = self.cls_token.unsqueeze(0).unsqueeze(0) # (1, 1)

        # generar hasta max_length o hasta sep token
        while (target_pred.shape[1] < max_length) and (target_pred[0, -1] != self.sep_token):
            scores = self.forward(input, target_pred) # (1, target_seq_length, vocab_size)

            next_token = scores[:, -1, :].argmax(dim=-1, keepdim=True) # (1, 1)
            target_pred = torch.cat([target_pred, next_token], dim=1)

    def loss(self, input, target):
        return None
