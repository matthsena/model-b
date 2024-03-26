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
    inception_dict = {k: v for k, v in inception_preds}
    resnet50_dict = {k: v for k, v in resnet50_preds}
    summed_values = Counter(inception_dict)
    summed_values.update(Counter(resnet50_dict))
    summed_values = {k: round(v * 50, 2) for k, v in summed_values.items()}
    result = sorted(summed_values.items(),
                    key=lambda x: x[1], reverse=True)[:2]
    return [f'{k}' for k, _ in result]


def discart_or_format_ocr(ocr_list):
    if not ocr_list:
        return []
    result = [item for item in ocr_list if item[1] > 0.5]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [f'{item[0]} ({round(item[1], 2)}%)' for item in result]


def format_panoptic_list(panoptic_list):
    counts = Counter(item for sublist in panoptic_list for item in sublist)
    result = [f'{count} {item}' for item, count in counts.items()]
    return result


def generate_text(preds1, preds2, panoptic1, panoptic2, ocr1, ocr2):
    messages = [
        {"role": "user", "content": f"""
        Given characteristics extracted from two different scenes, if Scene 1 is the same as Scene 2, JUST response with `yes` otherwise response with `no`.:
        Scene 1: We assume that the scene could be {preds1[0]} or {preds1[1]}, the objects present are {panoptic1}.
        Scene 2: We assume that the scene could be {preds2[0]} or {preds2[0]}, the objects present are {panoptic2}.
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
