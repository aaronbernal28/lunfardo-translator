from datasets import load_dataset
from codebase.utils import genai_samples, load_client, save_samples
from google import genai

# Cargamos el dataset para el par en-es_MX 998 muestras
dataset = load_dataset("google/wmt24pp", "en-es_MX", split="train")

# Seteamos la API de GEMINI
client = load_client()

def obtener_listas_oraciones_es(dataset):
    oraciones_es = []
    for data in dataset:
        if not data['is_bad_source']:
            oraciones_es.append(data['target'])
    return oraciones_es

oraciones_es = obtener_listas_oraciones_es(dataset)
oraciones_es, oraciones_es_lf = genai_samples(oraciones_es, client)

# guardamos en un archivo de texto
save_samples(oraciones_es, oraciones_es_lf, "data/es_MX-es_LF_1000.txt")