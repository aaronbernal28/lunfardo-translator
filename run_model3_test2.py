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

data = pd.read_csv('data/es-es_LF_100k.txt', sep='\t', header=None, on_bad_lines='skip')
filtered_vocab = np.load('data/sub_vocab.npy').tolist()
data.columns = ['spanish', 'lunfardo']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data['spanish'], data['lunfardo'],
                                                  test_size=0.1, random_state=28)

train_dataset = dataset_token(X_train, y_train)
val_dataset = dataset_token(X_val, y_val)

# Inicializa el modelo 
model = model3(
    name = 'model3_test_plot',
    filtered_vocab=filtered_vocab,
    d_model=512//2,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024
).to(device)

print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")

BATCH_SIZE = 10
MAX_STEPS = 500
LR = 1e-3

data = data.head(BATCH_SIZE)
dataset_overfit = dataset_token(data['spanish'], data['lunfardo'])
loader_overfit = DataLoader(dataset_overfit, batch_size=BATCH_SIZE, collate_fn=custom_collate)
_, _ = train(model, loader_overfit, max_steps=MAX_STEPS, lr=LR, verbose_each=10, plot=True)