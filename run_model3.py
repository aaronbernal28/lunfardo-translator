import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from codebase.models.model3 import model3
from codebase.train import dataset_token, custom_collate, train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
torch.cuda.set_per_process_memory_fraction(0.9)
PER_DEVICE_BATCH = 16 # batch size en memoria GPU

MODEL = model3
BATCH_SIZE = 16 # efectivo batch size
MAX_STEPS = 10000
LR = 1e-3
D_MODEL = 512//2
N_HEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024
NAME = f'{MODEL.__name__}_D{D_MODEL}_H{N_HEAD}_E{NUM_ENCODER_LAYERS}_D{NUM_DECODER_LAYERS}_F{DIM_FEEDFORWARD}_BS{BATCH_SIZE}_MS{MAX_STEPS}_LR{LR}'

data = pd.read_csv('data/es-es_LF_100k.txt', sep='\t', header=None, on_bad_lines='skip')
filtered_vocab = np.load('data/sub_vocab.npy').tolist()
data.columns = ['sp', 'lf']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data['sp'], data['lf'], test_size=0.05, random_state=28)

train_dataset = dataset_token(X_train, y_train)
val_dataset = dataset_token(X_val, y_val)

# Inicializa el modelo 
model = MODEL(
    name = NAME,
    filtered_vocab=filtered_vocab,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD
).to(device)

print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")

train_dataloader = DataLoader(train_dataset, batch_size=PER_DEVICE_BATCH, collate_fn=custom_collate)
val_dataloader = DataLoader(val_dataset, batch_size=PER_DEVICE_BATCH, collate_fn=custom_collate)

_, _ = train(model, train_dataloader, val_dataloader, batch_size = BATCH_SIZE, max_steps=MAX_STEPS, lr=LR, verbose_each=MAX_STEPS//6, perplexity=True, interactive_plot=False)