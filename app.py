import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from itertools import combinations, product

from models.cv.vgg19 import FeatureExtractor
from models.cv.ocr import OCRPredictor
from models.cv.inception3 import predict as inception3_predictor
from models.cv.resnet50 import predict as resnet50_predictor
from models.cv.panoptic import PanopticPredictor
import models.llm.zero_shot as LLMZeroShot
import utils.prompt as prompt
from utils.scores import ScoreCalculator
import matplotlib.pyplot as plt
import concurrent.futures
import database.sqlite as database
import time

class ImgFeatureExtractor:
    def __init__(self, dir):
        self.dir = dir
        self.extractor = FeatureExtractor()
        self.folders = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]
        self.db = database.SQLiteOperations()
        self.ocr_predictor = OCRPredictor()
        self.panoptic_predictor = PanopticPredictor()
        self.zero_shot = LLMZeroShot.LLMZeroShot()
    

    def get_img_files(self, folder, path):
        folder_path = os.path.join(folder, path)
        file_list = [file for file in os.listdir(
            folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        return (path, sorted(file_list))

    def extract_features(self, base_folder, img_files, folder):
        features = {}
        for lang, photos in img_files:
            for photo in photos:
                img_path = os.path.join(lang, photo)
                img = os.path.join(base_folder, img_path)
                features[img_path] = self.extractor.extract_features(img)
                # Salva no DB e pega outros itens
                features_list = features[img_path].tolist()
                feature_history = self.db.select_by_features(features_list)                
                if feature_history is not None:
                    self.db.upsert(img, lang, folder, features_list, feature_history['ocr'], feature_history[
                            'panoptic'], feature_history['inception_v3'], feature_history['resnet50'])
                    print(f"Found features in history for {img}")
                else:
                    with ThreadPoolExecutor() as executor:
                        img_cv2 = cv2.imread(img)
                        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                        ocr_texts_future = executor.submit(
                            self.ocr_predictor.predict, img_cv2)
                        inception_classes_future = executor.submit(
                            inception3_predictor, img_cv2)
                        resnet_classes_future = executor.submit(
                            resnet50_predictor, img_cv2)
                        panoptic_classes_future = executor.submit(
                            self.panoptic_predictor.predict, img_cv2)

                    ocr_texts = ocr_texts_future.result()
                    inception_classes = inception_classes_future.result()
                    resnet_classes = resnet_classes_future.result()
                    panoptic_classes = panoptic_classes_future.result()

                    self.db.upsert(img, lang, folder, features_list, ocr_texts,
                            panoptic_classes, inception_classes, resnet_classes)

        return features

    def get_comparison_list(self, img_files):
        compare_list = []
        seen = set()
        for (lang1, imgs1), (lang2, imgs2) in combinations(img_files, 2):
            for img1, img2 in product(imgs1, imgs2):
                items = tuple(sorted([lang1, lang2, img1, img2]))
                if items not in seen:
                    seen.add(items)
                    compare_list.append(((lang1, lang2), (img1, img2)))
        return compare_list

    def get_results(self, base_folder, compare_list, features):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.extractor.compare_features, features[os.path.join(lang1, img1)], features[os.path.join(
                lang2, img2)]): (lang1, lang2, img1, img2) for (lang1, lang2), (img1, img2) in compare_list}
            for future in concurrent.futures.as_completed(futures):
                lang1, lang2, img1, img2 = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('[APP] (%r) Erro no app: %s' % (lang1, exc))
                else:
                    results.append({
                        'article': base_folder,
                        'original': lang1,
                        'compare': lang2,
                        'original_photo': img1,
                        'compare_photo': img2,
                        'distance': result
                    })
        return results

    def plot_bar(self, dfs, titles, article):
        _, axes = plt.subplots(nrows=1, ncols=len(dfs), figsize=(15, 5))
        for df, ax, title in zip(dfs, axes, titles):
            df.plot(kind='bar', ax=ax, legend=True)
            mean = df.mean().mean()
            median = df.stack().median()
            ax.axhline(mean, color='red', linestyle='--', label='Mean')
            ax.axhline(median, color='blue', linestyle='-.', label='Median')
            ax.set_ylabel('Score')
            ax.set_xlabel('Langs')
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.tight_layout()
        plt.savefig(f'plots/{article}.png')

    def generate_plots(self, df):
        articles = df.article.unique()
        articles = np.array(articles)
        articles = np.sort(articles)
        for article in articles:
            df_article = df[df['article'] == article]
            df_simple = df_article[df_article['type'] == 'simple']
            df_simple = df_simple.drop(columns=['type', 'article'])
            df_decay = df_article[df_article['type'] == 'decay']
            df_decay = df_decay.drop(columns=['type', 'article'])
            dfgrowth = df_article[df_article['type'] == 'growth']
            dfgrowth = dfgrowth.drop(columns=['type', 'article'])
            df_parabola = df_article[df_article['type'] == 'parabola']
            df_parabola = df_parabola.drop(columns=['type', 'article'])
            self.plot_bar([df_simple, df_decay, dfgrowth, df_parabola], [
                     'simple', 'decaimento', 'growth', 'parabola'], article)
    
    def check_with_llm(self, df, folder):
        all_imgs = self.db.select_by_article(folder)
        
        # Using a loop to process each row
        for _, row in df.iterrows():
            try:
                path = f'{self.dir}/{folder}/{row["languages"][0]}/{row["original_photo"]}'
                current_img = self.db.select_by_path(path)
                found_langs = []
                
                if current_img is not None:  # Add this check
                    for img in all_imgs:
                        if img['file_path'] != path and img['lang'] != row['languages'][0] and img['article'] == folder and img['lang'] not in found_langs:
                            try:
                                preds1 = prompt.merge_and_sum_predictions(current_img['inception_v3'], current_img['resnet50'])
                                preds2 = prompt.merge_and_sum_predictions(img['inception_v3'], img['resnet50'])
                                panoptic1 = prompt.format_panoptic_list(current_img['panoptic'])
                                panoptic2 = prompt.format_panoptic_list(img['panoptic'])
                                ocr1 = prompt.discart_or_format_ocr(current_img['ocr'])
                                ocr2 = prompt.discart_or_format_ocr(img['ocr'])
                                llm_output = self.zero_shot.generate_text(preds1, preds2, panoptic1, panoptic2, ocr1, ocr2, path, img['file_path'])
                            except Exception as e:
                                print(f"Error while processing predictions: {e}")
                                continue
                            
                            if llm_output is None:
                                is_similar = False
                            else:
                                is_similar = bool(re.search(r'yes', llm_output, re.IGNORECASE))
                            
                            if is_similar:
                                found_langs.append(img['lang'])
                                if img['lang'] not in row['languages']:
                                    row['languages'].append(img['lang'])
                                    row['num_languages'] = len(row['languages'])
                                
            except Exception as e:
                print(f"Error while processing row: {e}")

            # Access the "data" variable here
            print(f'Processed {row["original_photo"]}')
        
        # Save the DataFrame after processing all rows
        try:
            df.to_csv(f'output/{folder}-zero_shot.csv', index=False)
        except Exception as e:
            print(f"Error while saving DataFrame: {e}")

    def run(self):
        start = time.time()
        final_df = pd.DataFrame()
        langs = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']
        for folder in self.folders:
            base_path = f'{self.dir}/{folder}/'
            img_files = list(map(lambda path: self.get_img_files(
                base_path, path), langs))
            features = self.extract_features(base_path, img_files, folder)
            compare_list = self.get_comparison_list(img_files)
            results = self.get_results(folder, compare_list, features)
            df_result = pd.DataFrame(results)
            score = ScoreCalculator()
            score_z, uniques = score.calculate_score(df_result, folder)
            self.check_with_llm(uniques, folder)
            df = pd.DataFrame(score_z)
            final_df = pd.concat([final_df, df])
        self.generate_plots(final_df)
        end = time.time()
        elapsed = (end - start) // 60
        print(
            f"[APP]: {elapsed} minutos e {(end - start) % 60:.2f} segundos")


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_dir = f'images/{dataset}'
    extractor = ImgFeatureExtractor(data_dir)
    extractor.run()
