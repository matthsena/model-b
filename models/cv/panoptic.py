from collections import Counter
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation, logging
import json
from torch.amp import autocast

logging.set_verbosity_error()


class PanopticPredictor:
    def __init__(self, coco_classes_path='resources/coco_2017_cat.json'):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50-panoptic")
        self.model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic", ignore_mismatched_sizes=True)

        with open(coco_classes_path) as f:
            self.coco_classes = json.load(f)

    def predict(self, image):
        try:
            with torch.no_grad(), autocast('cuda'):
                inputs = self.feature_extractor(
                    images=image, return_tensors="pt")
                outputs = self.model(**inputs)

                processed_sizes = torch.as_tensor(
                    inputs["pixel_values"].shape[-2:]).unsqueeze(0)
                result = self.feature_extractor.post_process_panoptic(
                    outputs, processed_sizes, threshold=0.5)[0]  # Adjust threshold as needed
            del inputs, outputs
            torch.cuda.empty_cache()

            extracted_classes = []
            for info in result['segments_info']:
                category_id = str(info['category_id'])
                if info["isthing"]:
                    cat_id = self.coco_classes['things_ids'].get(category_id)
                    if cat_id is not None:
                        extracted_classes.append(
                            self.coco_classes['things'][cat_id])
                else:
                    stuff_id = self.coco_classes['stuffs_ids'].get(category_id)
                    if stuff_id is not None:
                        extracted_classes.append(
                            self.coco_classes['stuffs'][stuff_id])

            counts = Counter(extracted_classes)
            return [(key, value) for key, value in counts.items()]
        except Exception as e:
            print(f"An error occurred on panoptic_predictor: {e}")
            return None
