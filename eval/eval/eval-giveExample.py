import csv
import json
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import pandas as pd

from infer import infer


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def print_2d_list(matrix):
    output = []
    # Number of columns
    cols = len(matrix[0])

    for col in range(cols):
        for row in matrix:
            # Convert each element to string before appending
            output.append(str(row[col]))

    # Join and return the string representation
    return "\n".join(output)


if __name__ == "__main__":
    dataset = "mteb_emotion"
    num_classes = 5  # update with the number of classes you have

    # visualize
    threshold = 0.90

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_classes, 2, figsize=(20, 6 * num_classes))

    results = []

    for class_idx in range(num_classes):
        file_path_class = os.path.join('..', 'score', dataset, f'class{class_idx}', f'mask_class{class_idx}.json')
        loaded_predicted_class = load_json(file_path_class)
        data = [score[2] for score in loaded_predicted_class]

        # Extract indices based on threshold from JSON data
        filtered_indices = [item[0] for item in loaded_predicted_class if item[2] < threshold]

        # Print data from CSV based on these indices
        output = []
        csv_df = pd.read_csv(f"../data/{dataset}/train.csv", index_col=0)
        filtered_data = csv_df[csv_df.index.isin(filtered_indices)]
        pd.set_option('display.max_colwidth', None)

        for _, row in filtered_data.iterrows():
            print(_, row['text'], row['label'])
            print('-' * 80)  # Optional: print a separator for clarity







