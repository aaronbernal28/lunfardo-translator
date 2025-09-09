import numpy as np
import os
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
import time

def train(model, train_loader, val_loader=None, epoch_max=100, lr=1e-3):
    train_losses = []
    val_losses = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Iniciando entrenamiento...")
    print("-" * 50)
    start_time = time.time()
    for epoch in range(epoch_max):
        # Entrenamiento
        model.train()
        train_loss = 0
        n = 0

        for _, batch in enumerate(train_loader):
            input = batch['input']
            target = batch['target']

            optimizer.zero_grad()

            loss = model.loss(input, target)

            loss.backward()

            # Gradient clipping para evitar exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        train_losses.append(train_loss)

        # Validación
        if val_loader is not None:
            model.eval()
            val_loss = 0
            m = 0
            with torch.no_grad():
                for _, batch in enumerate(val_loader):
                    input = batch['input']
                    target = batch['target']

                    loss = model.loss(input, target)
                    val_loss += loss.item()
                    m += 1
            val_loss /= m
            val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            if val_loader is None:
                print(f'Época {epoch+1}/{epoch_max}: Pérdida Entrenamiento: {train_loss:.4f}', end='\r')
            else:
                print(f'Época {epoch+1}/{epoch_max}: Pérdida Entrenamiento: {train_loss:.4f} | Pérdida Validación: {val_loss:.4f}', end='\r')
    
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    return train_losses, val_losses