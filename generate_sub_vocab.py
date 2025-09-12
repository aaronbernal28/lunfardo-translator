import pandas as pd
import numpy as np
from codebase.train import tokenizer

data = pd.read_csv('data/es-es_LF_100k.txt', sep='\t', header=None, on_bad_lines='skip')
data.columns = ['spanish', 'lunfardo']

from collections import Counter

def get_top_tokens_with_specials(row1, row2, tokenizer, top_k=10000):
    token_counter = Counter()

    # Count tokens in both datasets
    for txt in list(row1):
        if isinstance(txt, str):
            tokens = tokenizer.encode(txt, truncation=True)
            token_counter.update(tokens)

    for txt in list(row2):
        if isinstance(txt, str):
            tokens = tokenizer.encode(txt, truncation=True)
            token_counter.update(tokens)

    # Get top_k tokens
    most_common_tokens = token_counter.most_common(top_k)
    sorted_selected = sorted([token_id for token_id, _ in most_common_tokens])

    return [0] + sorted_selected

sub_vocab = get_top_tokens_with_specials(data['spanish'], data['lunfardo'], tokenizer)

np.save('data/sub_vocab.npy', np.array(sub_vocab))

