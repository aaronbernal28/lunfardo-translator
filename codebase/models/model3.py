import torch
import torch.nn as nn
import torch.nn.functional as F
#from codebase import utils as ut
from transformers import BertModel
from torcheval.metrics.text import Perplexity
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

class model3(nn.Module):
    def __init__(self, name, filtered_vocab,
                 d_model=512, 
                 nhead=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        '''
        modelo basado en BERT + transformer + filtered vocabulary
        '''
        super().__init__()
        self.name = name
        self.pad_token = torch.tensor(0).to(device)
        self.sep_token = torch.tensor(102).to(device)
        self.cls_token = torch.tensor(101).to(device)

        # cargar embeddings de BERT
        self.embeddings = BertModel.from_pretrained("bert-base-multilingual-uncased").to(device).eval().embeddings.word_embeddings.weight.detach()
        torch.cuda.empty_cache()
        self.vocab = torch.tensor(filtered_vocab).to(device)

        self.vocab_size = self.vocab.shape[0]
        self.emb_dim = self.embeddings.shape[1]
        self.d_model = d_model

        # proyectar embeddings a d_model
        if self.d_model != self.emb_dim:
            self.linear_in = nn.Linear(self.emb_dim, d_model)

        self.transformer = nn.Transformer(d_model=self.d_model,
                                          batch_first=True,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input, target):
        '''
        input: (batch_size, input_seq_length)
        target: (batch_size, target_seq_length)

        input_emb: (batch_size, input_seq_length, emb_dim)
        target_emb: (batch_size, target_seq_length, d_model)

        scores: (batch_size, target_seq_length, vocab_size)
        '''
        input_emb = self.embeddings[input]
        target_emb = self.embeddings[target]

        # target_mask evita que el transformer vea futuros tokens para la generacion del target
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_emb.size(1)).to(target_emb.device, dtype=torch.bool)

        if self.d_model != self.emb_dim:
            input_emb = self.linear_in(input_emb)
            target_emb = self.linear_in(target_emb)
            
        scores = self.transformer.forward(src=input_emb, tgt=target_emb,
                                          tgt_mask=target_mask,
                                          src_key_padding_mask=(input == self.pad_token.item()).bool(),
                                          tgt_key_padding_mask=(target == self.pad_token.item()).bool()) # (batch_size, target_seq_length, d_model)
        scores = self.linear(scores)
        return scores
    
    def generate(self, input, max_length=512):
        '''
        input: (input_seq_length,)
        target_pred: (generated_seq_length,)
        '''
        with torch.no_grad():
            input = input.unsqueeze(0) # (1, input_seq_length)
            target_pred = self.cls_token.unsqueeze(0).unsqueeze(0) # (1, 1)
    
            # proyectar embeddings a d_model
            input_emb = self.embeddings[input]
            if self.d_model != self.emb_dim:
                input_emb = self.linear_in(input_emb)
    
            # esto para se mantiene constante
            encoder_output = self.transformer.encoder(input_emb,
                                                      src_key_padding_mask=(input == self.pad_token.item()).bool())
    
            for _ in range(max_length - 1):
                target_emb = self.embeddings[target_pred]
    
                # proyectar embeddings a d_model
                if self.d_model != self.emb_dim:
                    target_emb = self.linear_in(target_emb)
    
                target_mask = nn.Transformer.generate_square_subsequent_mask(target_emb.size(1)).to(target_emb.device, dtype=torch.bool)
    
                # esa el encoder_output calculado
                decoder_output = self.transformer.decoder(
                    target_emb, encoder_output,
                    tgt_mask=target_mask,
                    tgt_key_padding_mask=(target_pred == self.pad_token.item()).bool()
                ) # (batch_size, target_seq_length, d_model)
    
                scores = self.linear(decoder_output) # (batch_size, target_seq_length, self.vocab_size)
                next_token = scores[:, -1, :].argmax(dim=-1, keepdim=True)
                next_token = self.get_original_token(next_token)
                target_pred = torch.cat([target_pred, next_token], dim=1) # dim(target_pred) += (0, 1)
    
                if next_token.item() == self.sep_token.item():
                    break
                
            return target_pred.squeeze(0)
    
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
        '''
        input: (batch_size, input_seq_length)
        target: (batch_size, target_seq_length)
        '''
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        scores = self.forward(input, target[:, :-1]) # no incluir el token final para predecir la siguiente palabra
        predictions = scores.reshape(-1, self.vocab_size) # (batch_size * pred_seq_length, vocab_size)
        targets = target[:, 1:].reshape(-1) # (batch_size * target_seq_length,)

        limited_targets = self.get_limited_token(targets)
        limited_pad_idx = self.get_limited_token(self.pad_token.unsqueeze(0)).item()
    
        # Cross-entropy loss
        loss = F.cross_entropy(predictions, limited_targets, ignore_index=limited_pad_idx)
        return loss
    
    def get_original_token(self, limited_tokens):
        '''
        limited_tokens: tensor() (batch_size, seq_len)
            0 <= token <= self.vocab.shape[0] ~ 10k
        output : tensor() (batch_size, seq_len) 
        '''
        return self.vocab[limited_tokens]
    
    def get_limited_token(self, tokens):
        '''
        tokens: tensor() (batch_size, seq_len)
            0 <= token <= self.embeddings.shape[0] ~ 100k
        output: tensor() (batch_size, seq_len)
        Ejemplo:
            >>> vocab = torch.tensor([101, 102, 106, 112, 117, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 136, 142, 143]) # falta el token PAD
            >>> tokens = torch.tensor([[106, 102, 101], [117, 112, 101]])
            >>> torch.searchsorted(vocab, tokens)
            tensor([[2, 1, 0],
                    [4, 3, 0]])
        '''
        res = torch.searchsorted(self.vocab, tokens)
        res = torch.clamp(res, 0, self.vocab_size - 1) # solucion provisoria
        return res
    
    def perplexity(self, val_loader: DataLoader, metric: Perplexity):
        '''
        Ejecutar en modo evaluacion (with torch.no_grad())
        '''
        for _, batch in enumerate(val_loader):
            input = batch['input']
            target = batch['target']

            if target.dim() == 1:
                target = target.unsqueeze(0)
            if input.dim() == 1:
                input = input.unsqueeze(0)

            scores = self.forward(input, target[:, :-1]) # no incluir el token final para predecir la siguiente palabra
            predictions = scores.reshape(-1, self.vocab_size) # (batch_size * pred_seq_length, vocab_size)
            targets = target[:, 1:].reshape(-1) # (batch_size * target_seq_length,)
            limited_targets = self.get_limited_token(targets)
        
            metric.update(predictions, limited_targets)

        res = metric.compute()
        metric.reset()
        return res