import pandas as pd
from codebase.utils import load_client, save_samples, genai_samples_parallel
from google import genai
from sklearn.model_selection import train_test_split

# Cargamos el dataset para el par en-es_MX 998 muestras
dataset = pd.read_csv('data/es-gl.txt', sep='\t', header=None, on_bad_lines='skip')
dataset.columns = ['spanish', 'gl']

# Filtrado del dataset original
dataset = dataset[-dataset['gl'].isna()]
len_words_per_sentence = dataset['spanish'].apply(lambda txt: len(txt.split(' ')))
dataset = dataset[(len_words_per_sentence >= 2) & (len_words_per_sentence <= 512/8)]

# Sample0 500k
#_, dataset_sampled = train_test_split(dataset, test_size=100000, random_state=28)

## Para obtener el complemento de dataset_sampled
dataset_sampled, _ = train_test_split(dataset, test_size=100000, random_state=28)

# Seteamos la API de GEMINI
client = load_client()
hundredk = int(1e5)
for i in range(hundredk, int(1e6)+hundredk, hundredk):
    oraciones_es = dataset_sampled['spanish'].tolist()
    oraciones_es = oraciones_es[i-hundredk:i]
    oraciones_es, oraciones_es_lf = genai_samples_parallel(oraciones_es, client, num_workers=190)

    # guardamos en un archivo de texto
    save_samples(oraciones_es, oraciones_es_lf, "data/es-es_LF_1M.txt", overwrite=False)