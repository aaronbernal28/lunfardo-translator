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

        loss = model.loss(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

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

def train_steps_perplexity(model, train_loader, val_loader, max_steps=1000, lr=1e-3, verbose_each=50, batch_size=32):
    """
    Basado en step - plot interactivo cada verbose_each steps
    Implementa un gradient accumulation
    """
    if val_loader is None:
        return False
    
    assert batch_size >= 16 and batch_size % 16 == 0, "requiere batch_size >= 16 y divisible por 16"
    accumulation_steps = batch_size // 16

    perplexities_xaxis = []
    perplexities = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    metric = Perplexity(ignore_index=model.get_limited_token(model.pad_token.unsqueeze(0)).item()).to(device)
    start_time = time.time()
    
    train_iterator = iter(train_loader)
    step = 0

    plt.ion()
    fig, ax = plt.subplots()
    curve_perplexity = None

    while step < max_steps+1:
        model.train()
        accumulated_loss = 0
        
        # Gradient accumulation loop
        for acc_step in range(accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                # reinicia la iteracion por los batches
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            input = batch['input']
            target = batch['target']

            loss = model.loss(input, target)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Validacion
        if step % verbose_each == 0:
            print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {loss.item():.4f}', end='\r')
            
            model.eval()
            with torch.no_grad():
                perplexity_val = model.perplexity(val_loader, metric).item()
                perplexities.append(perplexity_val)
                perplexities_xaxis.append(step)

            if curve_perplexity is None:
                curve_perplexity, = ax.plot(perplexities_xaxis, perplexities, "o-")
                ax.set_xlabel('Steps')
                ax.set_ylabel('Perplexity')
                ax.set_xlim((0, max_steps))
                ax.grid(True)
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

    box_text = f'Perplexity final: {perplexities[-1]:.2f}\nExecution time: {(end_time - start_time)/60:.2f} minutos'
    ax.text(0.7, 0.9, box_text, transform=ax.transAxes)
    
    plt.ioff()
    plt.savefig(f'images/{model.name}_perplexity.png')
    plt.show()

    return perplexities_xaxis, perplexities

def train_steps_perplexity_static(model, train_loader, val_loader, max_steps=1000, lr=1e-3, verbose_each=50, batch_size=32):
    """
    Basado en step - genera plot al final del entrenamiento
    Implementa un gradient accumulation
    """
    if val_loader is None:
        return False
    
    assert batch_size >= 16 and batch_size % 16 == 0, "requiere batch_size >= 16 y divisible por 16"
    accumulation_steps = batch_size // 16

    perplexities_xaxis = []
    perplexities = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    metric = Perplexity(ignore_index=model.get_limited_token(model.pad_token.unsqueeze(0)).item()).to(device)
    start_time = time.time()
    
    train_iterator = iter(train_loader)
    step = 0

    while step < max_steps+1:
        model.train()
        accumulated_loss = 0
        
        # Gradient accumulation loop
        for acc_step in range(accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                # reinicia la iteracion por los batches
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            input = batch['input']
            target = batch['target']

            loss = model.loss(input, target)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Validacion
        if step % verbose_each == 0:
            model.eval()
            with torch.no_grad():
                perplexity_val = model.perplexity(val_loader, metric).item()
                perplexities.append(perplexity_val)
                perplexities_xaxis.append(step)
                print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {accumulated_loss:.4f} | Perplexity = {perplexity_val:.4f}', end='\r')

        step += 1

    torch.cuda.empty_cache()
    end_time = time.time()

    # Create and save plot at the end
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(perplexities_xaxis, perplexities, "o-", linewidth=2, markersize=6)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Perplexity')
    ax.set_title(f'{model.name} - Validation Perplexity')
    ax.grid(True, alpha=0.3)
    
    # Add final stats box
    box_text = f'Perplexity final: {perplexities[-1]:.2f}\nExecution time: {(end_time - start_time)/60:.2f} minutos\nBatch size: {batch_size}'
    ax.text(0.7, 0.9, box_text, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'images/{model.name}_perplexity.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nEntrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    return perplexities_xaxis, perplexities

def train(model, train_loader, val_loader=None, batch_size=16, max_steps=1000, lr=1e-3, verbose_each=50, perplexity=True, interactive_plot=False):
    if perplexity:
        if interactive_plot:
            out1, out2 = train_steps_perplexity(model, train_loader, val_loader, max_steps=max_steps, lr=lr, verbose_each=verbose_each, batch_size=batch_size)
        else:
            out1, out2 = train_steps_perplexity_static(model, train_loader, val_loader, max_steps=max_steps, lr=lr, verbose_each=verbose_each, batch_size=batch_size)
    else:
        out1, out2 = train_steps(model, train_loader, val_loader, max_steps=max_steps, lr=lr, verbose_each=verbose_each)
    torch.save(model.state_dict(), f'checkpoints/{model.name}.pth')
    return out1, out2
