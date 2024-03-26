from transformers import AutoTokenizer, pipeline
import torch
from collections import Counter

model = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def merge_and_sum_predictions(inception_preds, resnet50_preds):
    summed_values = Counter(dict(inception_preds))
    summed_values.update(Counter(dict(resnet50_preds)))
    summed_values = {k: round(v * 50, 2) for k, v in summed_values.items()}
    result = sorted(summed_values.items(),
                    key=lambda x: x[1], reverse=True)[:2]
    return [f'{k} ({v}%)' for k, v in result]


def discart_or_format_ocr(ocr_list):
    result = [item for item in ocr_list if item[1] > 0.5]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [f'{item[0]} ({round(item[1], 2)}%)' for item in result]


def format_panoptic_list(panoptic_list):
    counts = Counter(panoptic_list)
    result = [f'{count} {item}' for item, count in counts.items()]
    return result


def generate_text(preds1, preds2, panoptic1, panoptic2, ocr1, ocr2):
    messages = [
        {"role": "user", "content": f"""
         Duas cenas podem nao ser visualmente iguais, mas podem transamitir a mesma mensagem.
         Por exemplo: uma cena com uma estatua em uma montanha com ceu aparente, considera-se similar a uma cena de uma estatua com uma floresta e pessoas em volta.
         Outro exemplo: uma cena com carros, predios e sinal de transito, considera-se similar a uma cena com carros, estrada e pessoas
         Mais um exemplo: uma cena com uma pessoa com oculos, considera-se similar a uma cena com uma pessoa sem oculos.

         Cena 1:
          Supomos que a cena pode ser {preds1[0]} ou {preds1[1]}, os objetos presentes sao {panoptic1} e os textos que aparecem na cena sao {ocr1}.

         Cena 2:
          Supomos que a cena pode ser {preds2[0]} ou {preds2[1]}, os objetos presentes sao {panoptic2} e os textos que aparecem na cena sao {ocr2}.

          Classifique essas duas cenas como similar ou nao similar.
        """},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95
    )
    return outputs[0]["generated_text"][len(prompt):]
