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
    dataset = "mteb_reviews"
    num_classes = 5  # update with the number of classes you have

    # visualize
    thresholds = [0.2, 0.4, 0.6, 0.8, 1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_classes, 2, figsize=(20, 6 * num_classes))

    results = []

    for class_idx in range(num_classes):
        file_path_class = os.path.join('..', 'score', dataset, f'class{class_idx}', f'mask_class{class_idx}.json')
        loaded_predicted_class = load_json(file_path_class)
        data = [score[2] for score in loaded_predicted_class]

        # Calculate percentages
        percentages = [(np.array(data) <= threshold).mean() * 100 for threshold in thresholds]

        results.append(percentages)

        # Visualize for data
        axs[class_idx][0].bar(thresholds, percentages, width=0.05)
        axs[class_idx][0].set_xlabel('Threshold')
        axs[class_idx][0].set_ylabel('Percentage (%)')
        axs[class_idx][0].set_title(f'Percentage of class {class_idx} lower than the threshold')
        axs[class_idx][0].grid(True)
        for i in range(len(thresholds)):
            axs[class_idx][0].text(thresholds[i], percentages[i], f'{percentages[i]:.5f}%', ha='center', va='bottom')

        # Plot the histogram
        axs[class_idx][1].hist(data, bins=30, edgecolor='black')  # you can adjust the number of bins as needed
        axs[class_idx][1].set_title(f'Data Distribution - Class {class_idx}')
        axs[class_idx][1].set_xlabel('Value')
        axs[class_idx][1].set_ylabel('Frequency')
        axs[class_idx][1].grid(True)

    print(print_2d_list(results))
    plt.tight_layout()
    plt.show()





