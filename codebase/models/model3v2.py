import torch
import torch.nn as nn
import torch.nn.functional as F
from codebase import utils as ut
from transformers import BertModel
from torcheval.metrics.text import Perplexity
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

class model3pro(nn.Module):
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
        self.pos_enc = ut.PositionalEncoding(d_model, dropout, 512).to(device).eval()
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
        self.linear_out = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input, target):
        '''
        input: (batch_size, input_seq_length)
        target: (batch_size, target_seq_length)

        output: (batch_size, target_seq_length, vocab_size)
        '''
        encoder_output = self.forward_encoder(input)
        output = self.forward_decoder(target, encoder_output)
        return output

    def forward_encoder(self, input):
        input_emb = self.embeddings[input]

        if self.d_model != self.emb_dim:
            input_emb = self.linear_in(input_emb)
        
        input_emb = self.pos_enc(input_emb)
        # esto para se mantiene constante
        output = self.transformer.encoder.forward(input_emb, is_causal=False)
        return output
    
    def forward_decoder(self, target, encoder_output):
        target_emb = self.embeddings[target]

        if self.d_model != self.emb_dim:
            target_emb = self.linear_in(target_emb)

        target_emb = self.pos_enc(target_emb)

        tgt_mask = self.transformer.generate_square_subsequent_mask(target_emb.size(1)).to(device)
        output = self.transformer.decoder.forward(target_emb, encoder_output,
                                                  tgt_is_causal=True,
                                                  tgt_mask=tgt_mask) # (batch_size, target_seq_length, d_model)

        return self.linear_out(output)

    def generate(self, input, max_length=512):
        '''
        input: (input_seq_length,)
        target_pred: (generated_seq_length,)
        '''
        with torch.no_grad():
            input = input.unsqueeze(0)  # (1, input_seq_length)

            encoder_output = self.forward_encoder(input)
            current_target = self.cls_token.unsqueeze(0).unsqueeze(0)  # (1, 1)
            for _ in range(max_length - 1):
                scores = self.forward_decoder(current_target, encoder_output)  # (batch_size, current_target_length, vocab_size)
                next_token = scores[:, -1, :].argmax(dim=-1, keepdim=True)
                next_token = self.get_original_token(next_token)
                current_target = torch.cat([current_target, next_token], dim=1)  # (1, current_target_length+1)

                if next_token.item() == self.sep_token.item():
                    break

            return current_target.squeeze(0)

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
        target: (batch_size, target_seq_length) - should include CLS at start and SEP at end
        '''
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        target_predictor = target[:, :-1]  # [CLS, word1, word2, ...] (except last token)
        target_labels = target[:, 1:]  # [word1, word2, ..., SEP] (except CLS)

        target_predicted = self.forward(input, target_predictor)
        target_predicted = target_predicted.reshape(-1, self.vocab_size)  # (batch_size * pred_seq_length, vocab_size)

        target_labels = target_labels.reshape(-1)  # (batch_size * pred_seq_length,)

        limited_targets = self.get_limited_token(target_labels)
        limited_pad_idx = self.get_limited_token(self.pad_token.unsqueeze(0)).item()

        # Cross-entropy loss
        loss = F.cross_entropy(target_predicted, limited_targets, ignore_index=limited_pad_idx)
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
        res = torch.searchsorted(self.vocab, tokens.contiguous())
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

            scores = self.forward(input, target[:, :-1])
            targets = target[:, 1:]
            limited_targets = self.get_limited_token(targets)
            metric.update(scores, limited_targets)

        res = metric.compute()
        metric.reset()
        return res