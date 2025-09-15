import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torcheval.metrics.text import Perplexity

model_name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dataset_token(Dataset):
    def __init__(self, x, y):
        self.pairs = list(zip(x, y))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        xs, ys = self.pairs[idx]
        tokenized_xs = torch.tensor(tokenizer.encode(xs, truncation=True)).to(device)
        tokenized_ys = torch.tensor(tokenizer.encode(ys, truncation=True)).to(device)

        return {
            'input': tokenized_xs,
            'target': tokenized_ys,
            'input_length': torch.tensor(tokenized_xs.shape[0]).to(device),
            'target_length': torch.tensor(tokenized_ys.shape[0]).to(device)
    }

def custom_collate(batch):
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)  # (batch, max_len, 768)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input': padded_inputs,
            'target': padded_targets,
            'input_length': torch.tensor([item['input_length'] for item in batch]),
            'target_length': torch.tensor([item['target_length'] for item in batch])}

def plot_losses(train_loss, val_loss, xlabel = 'Épocas'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    plt.xlabel(xlabel)
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_steps(model, train_loader, val_loader=None, max_steps=1000, lr=1e-3, verbose_each=50):
    """
    Basado en step - printea cada verbose_each steps
    """
    train_losses = []
    val_losses = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Iniciando entrenamiento...")
    print("-" * 50)
    start_time = time.time()
    
    train_iterator = iter(train_loader)
    if val_loader is not None:
        val_iterator = iter(val_loader)
    step = 0
    current_val_loss = 0
    while step < max_steps:
        model.train()
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            # reinicia la iteracion por los batches
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        input = batch['input']
        target = batch['target']

        optimizer.zero_grad()
        loss = model.loss(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        step += 1
        
        current_train_loss = loss.item()
        train_losses.append(current_train_loss)
        
        # Validacion
        if val_loader is not None:
            model.eval()
            
            with torch.no_grad():
                try:
                    batch_val = next(val_iterator)
                except StopIteration:
                    # reinicia la iteracion por los batches
                    val_iterator = iter(val_loader)
                    batch_val = next(val_iterator)
                input = batch_val['input']
                target = batch_val['target']
                loss_val = model.loss(input, target)
                current_val_loss = loss_val.item()
                val_losses.append(current_val_loss)
                
        # Progreso cada verbose_each
        if step % verbose_each == 0:
            if val_loader is not None:
                print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {current_train_loss:.4f} | Pérdida Validación: {current_val_loss:.4f}', end='\r')
            else:
                print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {current_train_loss:.4f}', end='\r')
    
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    
    return train_losses, val_losses

def train_steps_plot(model, train_loader, val_loader=None, max_steps=1000, lr=1e-3, verbose_each=50):
    """
    Basado en step - plot interactivo cada verbose_each steps
    """
    train_losses = []
    val_losses = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    
    train_iterator = iter(train_loader)

    if val_loader is not None:
        val_iterator = iter(val_loader)
    step = 0
    current_val_loss = 0

    plt.ion()
    fig, ax = plt.subplots()
    curve_train = None
    curve_val = None

    while step < max_steps:
        model.train()
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            # reinicia la iteracion por los batches
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        input = batch['input']
        target = batch['target']

        optimizer.zero_grad()
        loss = model.loss(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_train_loss = loss.item()
        train_losses.append(current_train_loss)
        
        # Validacion
        if val_loader is not None:
            model.eval()
            
            with torch.no_grad():
                try:
                    batch_val = next(val_iterator)
                except StopIteration:
                    # reinicia la iteracion por los batches
                    val_iterator = iter(val_loader)
                    batch_val = next(val_iterator)
                input = batch_val['input']
                target = batch_val['target']
                loss_val = model.loss(input, target)
                current_val_loss = loss_val.item()
                val_losses.append(current_val_loss)
                
        # Progreso cada verbose_each
        if step % verbose_each == 0:
            if curve_train is None:
                curve_train, = ax.plot(range(step+1), train_losses, label='Pérdida de Entrenamiento')
                if val_loader is not None:
                    curve_val, = ax.plot(range(step+1), val_losses, label='Pérdida de Validación')
                ax.set_xlabel('Steps')
                ax.set_ylabel('Loss')
                ax.set_xlim((0, max_steps))
                ax.grid(True)
                ax.legend()
            else:
                # Update data
                curve_train.set_xdata(range(len(train_losses)))
                curve_train.set_ydata(train_losses)
                if val_loader is not None and curve_val is not None:
                    curve_val.set_xdata(range(len(val_losses)))
                    curve_val.set_ydata(val_losses)
                
                # Recalculate limits and rescale
                ax.relim()
                ax.autoscale_view()
                
            # Force update
            fig.canvas.draw()
            fig.canvas.flush_events()
        step += 1
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    
    plt.ioff()
    plt.savefig(f'images/{model.name}.png')
    plt.show()
    return train_losses, val_losses

def train_steps_perplexity(model, train_loader, val_loader, max_steps=1000, lr=1e-3, verbose_each=50):
    """
    Basado en step - plot interactivo cada verbose_each steps
    """
    if val_loader is None:
        return False
    
    perplexities_xaxis = []
    perplexities = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    metric = Perplexity(ignore_index=model.get_limited_token(model.pad_token.unsqueeze(0)).item())
    start_time = time.time()
    
    train_iterator = iter(train_loader)
    step = 0

    plt.ion()
    fig, ax = plt.subplots()
    curve_perplexity = None

    while step < max_steps:
        model.train()
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            # reinicia la iteracion por los batches
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        input = batch['input']
        target = batch['target']

        optimizer.zero_grad()
        loss = model.loss(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validacion
        if step % verbose_each == 0:
            model.eval()
            with torch.no_grad():
                perplexity_val = model.perplexity(val_loader, metric).item()
                perplexities.append(perplexity_val)
                perplexities_xaxis.append(step)

            if curve_perplexity is None:
                curve_perplexity, = ax.plot(perplexities_xaxis, perplexities)
                ax.set_xlabel('Steps')
                ax.set_ylabel('Perplexity')
                ax.set_xlim((0, max_steps))
                ax.grid(True)
                ax.legend()
            else:
                curve_perplexity.set_xdata(perplexities_xaxis)
                curve_perplexity.set_ydata(perplexities)

                # Recalculate limits and rescale
                ax.relim()
                ax.autoscale_view()

            # Force update
            fig.canvas.draw()
            fig.canvas.flush_events()
        step += 1

    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    
    plt.ioff()
    plt.savefig(f'images/{model.name}_perplexity.png')
    plt.show()

    return perplexities_xaxis, perplexities

def train(model, train_loader, val_loader=None, max_steps=1000, lr=1e-3, verbose_each=50, perplexity=False, plot=False):
    if perplexity:
        out1, out2 = train_steps_perplexity(model, train_loader, val_loader, max_steps=1000, lr=1e-3, verbose_each=50)
    elif plot:
        out1, out2 = train_steps_plot(model, train_loader, val_loader, max_steps=max_steps, lr=lr, verbose_each=verbose_each)
    else:
        out1, out2 = train_steps(model, train_loader, val_loader, max_steps=max_steps, lr=lr, verbose_each=verbose_each)
    torch.save(model.state_dict(), f'checkpoints/{model.name}.pth')
    return out1, out2
