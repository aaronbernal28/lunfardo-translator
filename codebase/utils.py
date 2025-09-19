from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()

import re
import numpy as np

def preprocess_text(text, max_length_palabras=512*4):
    """Preprocesa el texto para limpieza básica"""
    # Limpiar pero mantener puntuación básica
    text = text.strip()
    # Convertir a minúsculas
    #text = text.lower()
    # Remover puntuación excesiva pero mantener puntos y comas
    text = re.sub(r'[^\w\sáéíóúüñ.,!¡?¿\']', '', text)
    # Remover espacios extra y truncar
    text = ' '.join(text.split()[:max_length_palabras])
    return text

def load_client():
    # Evito llamar al cliente varias veces
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

def genai_samples(samples: list[str], client: genai.Client):
    import time
    t0 = time.time()
    responses = []
    processed_samples = []
    skipped_count = 0
    query_in = os.getenv("QUERY_in")
    query_out = os.getenv("QUERY_out")
    for _sample in samples:
        _sample = preprocess_text(_sample)
        success = False
        
        # 3 intentos por muestra
        for attempt in range(3):
            try:
                response_es_lf = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=f'{query_in}:\n"{_sample}"\n{query_out}',
                    config=types.GenerateContentConfig(
                        temperature=0.75,
                        seed=28 + attempt, # Cambiar la semilla en cada intento de forma controlada
                        thinking_config=types.ThinkingConfig(thinking_budget=512)
                    )
                )
                
                # Check if we got valid text
                if response_es_lf.text is not None:
                    res = preprocess_text(response_es_lf.text)
                    responses.append(res)
                    processed_samples.append(_sample)
                    success = True
                    break
                else:
                    print(f"\nAttempt {attempt + 1}/3: Gemini returned None for sample (length: {len(_sample)})")
                    if attempt < 2:  # Don't sleep on the last attempt
                        time.sleep(1)
                        
            except Exception as e:
                print(f"\nAttempt {attempt + 1}/3: Error occurred: {e}")
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(1)
        
        # If all 3 attempts failed, skip this sample
        if not success:
            print(f"\nSkipping sample after 3 failed attempts: '{_sample[:50]}...'")
            skipped_count += 1

        # Progress in console
        total_processed = len(responses)
        total_samples = len(samples)
        elapsed_time = (time.time() - t0) / 60
        print(f"Processed {total_processed}/{total_samples} samples (skipped: {skipped_count}). Time elapsed: {elapsed_time:.2f} minutes", end='\r')

    elapsed_time = (time.time() - t0) / 60
    print(f"\nCompleted! Processed {len(responses)}/{len(samples)} samples (skipped: {skipped_count}). Total time: {elapsed_time:.2f} minutes")
    
    return processed_samples, responses

def save_samples(oraciones_es, oraciones_es_lf, filename, overwrite=True):
    mode = 'a'
    if overwrite:
        mode = 'w'
        
    with open(filename, mode, encoding="utf-8") as f:
        for oracion in zip(oraciones_es, oraciones_es_lf):
            f.write(oracion[0] + "\t" + oracion[1] + "\n")
    print(f"Samples saved to {filename}")


def genai_samples_parallel(samples: list[str], client: genai.Client, num_workers=4):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Dividir las muestras en lotes para cada trabajador
    batches = np.array_split(samples, num_workers)
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(genai_samples, batch.tolist(), client): batch for batch in batches}
        
        for future in as_completed(future_to_batch):
            try:
                processed_samples, responses = future.result()
                results.extend(zip(processed_samples, responses))
            except Exception as e:
                print(f"Error processing batch: {e}")

    # Desempaquetar resultados
    processed_samples, responses = zip(*results) if results else ([], [])
    
    return list(processed_samples), list(responses)

# Al parecer Transformer no tiene positional encoding por defecto :(
import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) - Add batch dimension here
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
# Example usage:

if __name__ == "__main__":
    sample_text = '''
    ¡Hola! ¿Cómo estás? Esto es una prueba de preprocesamiento de texto. 
    Vamos a limpiar este texto, pero mantener la puntuación básica, como comas y puntos. 
    Además, eliminaremos cualquier carácter extraño como @, #, $, %, ^, &, *, (, ), etc. 
    También NO convertiremos todo a minúsculas y eliminaremos espacios extra. ¡Vamos a ver cómo funciona esto!
    $$ %%% ### @@@ !!! ??? ... --- *** &&& ^^^ 1234567890
    '''
    print("Original:", sample_text)
    print("Preprocessed:", preprocess_text(sample_text, max_length_palabras=50))