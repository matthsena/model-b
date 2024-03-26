from collections import Counter

def merge_and_sum_predictions(inception_preds, resnet50_preds):
    inception_dict = {k: v for k, v in inception_preds}
    resnet50_dict = {k: v for k, v in resnet50_preds}
    summed_values = Counter(inception_dict)
    summed_values.update(Counter(resnet50_dict))
    summed_values = {k: round(v * 50, 2) for k, v in summed_values.items()}
    result = sorted(summed_values.items(),
                    key=lambda x: x[1], reverse=True)[:2]
    return [f'{k}' for k, _ in result]

def format_panoptic_list(panoptic_list):
    counts = Counter(item for sublist in panoptic_list for item in sublist)
    result = [f'{count} {item}' for item, count in counts.items()]
    return result


def discart_or_format_ocr(ocr_list):
    if not ocr_list:
        return []
    result = [item for item in ocr_list if item[1] > 0.5]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [item[0] for item in result]
