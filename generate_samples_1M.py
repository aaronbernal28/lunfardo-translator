import pandas as pd
from codebase.utils import load_client, save_samples, genai_samples_parallel
from google import genai
from sklearn.model_selection import train_test_split

# Cargamos el dataset
dataset = pd.read_csv('data/es-gl.txt', sep='\t', header=None, on_bad_lines='skip')
dataset.columns = ['spanish', 'gl']

# Filtrado del dataset original
dataset = dataset[-dataset['gl'].isna()]
len_words_per_sentence = dataset['spanish'].apply(lambda txt: len(txt.split(' ')))
dataset = dataset[(len_words_per_sentence >= 2) & (len_words_per_sentence <= 512/4)]
dataset_sampled, _ = train_test_split(dataset, test_size=100000, random_state=28) # (len(dataset) - 100000)

# Seteamos la API de GEMINI
client = load_client()
oraciones_es = dataset_sampled['spanish'].tolist()

_100k = int(1e5)
_1M = int(1e6)

# train 1M
for i in range(_100k, _1M + 1, _100k):
    print(f"Generating samples from {i-_100k} to {i} in oraciones_es of length {len(oraciones_es)}")
    oraciones_es_out, oraciones_es_lf = genai_samples_parallel(oraciones_es[i-_100k:i], client, num_workers=175)
    save_samples(oraciones_es_out, oraciones_es_lf, "data/main_1M_train.txt", overwrite=False)

# validation 50k
oraciones_es_out, oraciones_es_lf = genai_samples_parallel(oraciones_es[_1M: _1M + 50000], client, num_workers=175)
save_samples(oraciones_es_out, oraciones_es_lf, "data/main_1M_validation.txt", overwrite=False)

# test 10k
oraciones_es_out, oraciones_es_lf = genai_samples_parallel(oraciones_es[_1M + 50000: _1M + 50000 + 10000], client, num_workers=175)
save_samples(oraciones_es_out, oraciones_es_lf, "data/main_1M_test.txt", overwrite=False)