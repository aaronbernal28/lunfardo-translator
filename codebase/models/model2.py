import torch
import torch.nn as nn
import torch.nn.functional as F
from codebase import utils as ut
from transformers import BertModel

torch.cuda.empty_cache()

class model2(nn.Module):
    def __init__(self, name, device):
        '''
        modelo basado en BERT + transformer
        '''
        super().__init__()
        self.name = name
        self.device = device
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
        input: (input_seq_length,)
        returns: (generated_seq_length,)
        '''
        input = input.unsqueeze(0)  # (1, input_seq_length)
        target_pred = self.cls_token.unsqueeze(0).unsqueeze(0)  # (1, 1)

        # generar hasta max_length o hasta sep token
        for _ in range(max_length - 1):  # -1 porque comenzamos con CLS
            scores = self.forward(input, target_pred)  # (1, target_seq_length, vocab_size)
            next_token = scores[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            target_pred = torch.cat([target_pred, next_token], dim=1)

            if next_token.item() == self.sep_token.item():
                break
            
        return target_pred.squeeze(0) # (generated_seq_length,)
    
    def batch_generate(self, input_batch, max_length=512):
        """
        Lento pero deberia funcionar
        input_batch: (batch_size, input_seq_length)
        returns: list (generated_seq_length,) para cada elemento en el batch
        """
        results = []
        for i in range(input_batch.size(0)):
            generated = self.generate(input_batch[i], max_length)
            results.append(generated)
        return results
    
    def loss(self, input, target):
        scores = self.forward(input, target[:, :-1]) # no incluir el token final

        predictions = scores.reshape(-1, self.vocab_size) # (batch_size * pred_seq_length, vocab_size)

        targets = target[:, 1:].reshape(-1) # (batch_size * target_seq_length,)

        # Cross-entropy loss
        loss = F.cross_entropy(predictions, targets, ignore_index=self.pad_token.item())
        return loss