from paddleocr import PaddleOCR


class OCRPredictor:
    def __init__(self):
        self.model = PaddleOCR(use_angle_cls=True,
                               lang='en',
                               use_gpu=True,
                               det_algorithm='DB',
                               rec_algorithm='SVTR_LCNet',
                               det_db_score_mode='fast',
                               label_list=['0', '90', '180', '270'],
                               show_log=False)

    def predict(self, image):
        try:
            result = self.model.ocr(image, cls=True)
            filtered_values_and_prob = [
                item for _, item in result if item[1] >= 0.5]
            return filtered_values_and_prob
        except Exception as e:
            print(f"An error occurred in OCR fn: {str(e)}")
            return None
