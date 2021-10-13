import sys
sys.path.append('..')

import os
import pandas as pd

out_path = "../output"

model_names = ["roberta"]
dataset_nammes = ["scifact"]

for model in model_names:
    inspect_path = f"{out_path}/{model}"
    for dir in os.listdir(inspect_path):
        f = f"{inspect_path}/{dir}/feature.csv"
        try:
            predictions = list(pd.read_csv(f)['predicted_classes'])
        except Exception as e:
            print("?")
        count_0 = predictions.count(0)/sum(predictions)
        count_1 =predictions.count(1)/sum(predictions)
        print(f)
        print(f"class 0: {count_0}; class 1: {count_1}")