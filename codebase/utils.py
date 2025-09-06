from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()

def load_client():
    # Evito llamar al cliente varias veces
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

def genai_samples(samples: list[str], client: genai.Client) -> list[str]:
    import time
    t0 = time.time()
    responses = []
    for sample in samples:
        response_es_lf = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents='''
Translate to natural, hilarious and explicit lunfardo argentino, no waffle or other dialogue, same length:
"{}"
            '''.format(sample),
            config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=512)
            )
        )
        responses.append(response_es_lf.text)

        # Progreso en la consola
        print(f"Processed {len(responses)}/{len(samples)} samples. Time elapsed: {time.time() - t0:.2f} seconds", end='\r')
    return responses


def save_samples(oraciones_es, oraciones_es_lf, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for oracion in zip(oraciones_es, oraciones_es_lf):
            f.write(oracion[0] + "\t" + oracion[1] + "\n")
    print(f"Samples saved to {filename}")