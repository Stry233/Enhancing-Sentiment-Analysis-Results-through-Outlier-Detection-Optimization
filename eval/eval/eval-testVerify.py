import csv
import json
import os
import random

import numpy as np
import torch
import transformers
from matplotlib import pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import load_dataset
import pandas as pd


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_data(threshold, dataset, dataframe):
    mask = []
    class_num = 0

    # Iterate over class directories until no more are found
    while True:
        class_dir = os.path.join('..', 'score', dataset, f'class{class_num}')

        # If the directory doesn't exist, break the loop
        if not os.path.exists(class_dir):
            break

        file_path_class = os.path.join(class_dir, f'mask_class{class_num}.json')
        loaded_predicted_class = load_json(file_path_class)
        mask_class = [score[0] for score in loaded_predicted_class if score[2] <= threshold]
        mask.extend(mask_class)

        class_num += 1

    # Merge the masks and remove duplicates
    mask = list(set(mask))

    # Filter the DataFrame based on the index list
    filtered_df = dataframe.loc[mask].dropna()

    # Write the filtered DataFrame to a new CSV file
    data_dir = os.path.join("..", "data", dataset, f"{threshold}_train.csv")
    filtered_df.to_csv(data_dir, index=False)

    return data_dir


def start_test(train_data, test_data, mask, epoch=16, model_name="distilbert-base-uncased"):
    seed = 42
    # Fix the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load the data from CSV file
    dataset = load_dataset('csv', data_files={'train': train_data, 'test': test_data})
    num_classes = len(set(dataset['train']['label']))

    # Specify the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the data
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset['train']), num_proc=8)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        acc = accuracy_score(labels, preds)

        # compute confusion matrix and derive class-specific accuracy
        cm = confusion_matrix(labels, preds)
        class_accuracy = {}
        for i in range(cm.shape[0]):  # loop over each class
            class_accuracy[f'class_{i}_accuracy'] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0

        # calculate 'inlier accuracy' and 'outlier accuracy'
        inlier_labels = labels[mask]
        inlier_preds = preds[mask]
        inlier_accuracy = accuracy_score(inlier_labels, inlier_preds)

        # note: this assumes that 'mask' is a binary mask with the same shape as 'labels' and 'preds'
        outlier_mask = [i for i in range(len(labels)) if i not in mask]
        outlier_labels = labels[outlier_mask]
        outlier_preds = preds[outlier_mask]
        outlier_accuracy = accuracy_score(outlier_labels, outlier_preds) if len(outlier_mask) > 0 else 0

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'inlier_accuracy': inlier_accuracy,
            'outlier_accuracy': outlier_accuracy,
            **class_accuracy
        }

    # Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epoch,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=1e-06,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()

    # Evaluation
    evaluation_results = trainer.evaluate()

    # Get the losses per epoch
    logs = trainer.state.log_history

    # Print the evaluation results
    for key in sorted(evaluation_results.keys()):
        print(f"{key}: {evaluation_results[key]}")

    return logs


def load_data(dataset: str, split: str, percentage: float) -> pd.DataFrame:
    """Loads the specified dataset split."""
    if percentage == 1:
        percentage_prefix = ""
    else:
        percentage_prefix = str(percentage) + "_"
    return pd.read_csv(f"../data/{dataset}/{percentage_prefix}{split}.csv")


def generate_threshold_mask(dataset, threshold):
    mask = []
    class_num = 0

    # Iterate over class directories until no more are found
    while True:
        class_dir = os.path.join('..', 'score', dataset, f'class{class_num}')

        # If the directory doesn't exist, break the loop
        if not os.path.exists(class_dir):
            break

        file_path_class = os.path.join(class_dir, f'mask_class{class_num}.json')
        loaded_predicted_class = load_json(file_path_class)
        mask_class = [score[0] for score in loaded_predicted_class if score[2] <= threshold]
        mask.extend(mask_class)

        class_num += 1

    # Merge the masks and remove duplicates
    return list(set(mask))


if __name__ == "__main__":
    dataset = "mteb_counterfactual"
    portion = 1
    threshold_train = 0.4
    threshold_eval = 0.4
    epochs = 16

    # Load data
    ground_truth_train = load_data(dataset, "train", portion)
    ground_truth_test = load_data(dataset, "test", portion)

    # Generate data for threshold
    generate_data(threshold_train, dataset, ground_truth_train)

    # Test based on threshold given
    if portion == 1:
        percentage_prefix = ""
    else:
        percentage_prefix = str(portion) + "_"
    test_file_location = f"../data/{dataset}/{percentage_prefix}test.csv"
    train_file_location = f"../data/{dataset}/{threshold_train}_{percentage_prefix}train.csv"
    mask = generate_threshold_mask(dataset+"_eval", threshold_eval)

    labels = ground_truth_test['label']

    log = start_test(train_file_location, test_file_location, mask, epochs)

    inlier_accuracies = [epoch["eval_inlier_accuracy"] for epoch in log if "eval_inlier_accuracy" in epoch]
    outlier_accuracies = [epoch["eval_outlier_accuracy"] for epoch in log if "eval_outlier_accuracy" in epoch]
    eval_accuracies = [epoch["eval_accuracy"] for epoch in log if "eval_accuracy" in epoch]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 2), inlier_accuracies, label='Inlier Accuracy')
    plt.plot(range(1, epochs + 2), outlier_accuracies, label='Outlier Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Inlier and Outlier Accuracies over Epochs')
    plt.legend()
    plt.show()

    # Save the data to a CSV file
    with open(f"../imgs/analyze/testVerify_{dataset}_{threshold_train}_{threshold_eval}_analyze.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Inlier Accuracy", "Outlier Accuracy", "Weighted Accuracy"])
        for i, (in_acc, out_acc, eval_acc) in enumerate(zip(inlier_accuracies, outlier_accuracies, eval_accuracies)):
            writer.writerow([i + 1, in_acc, out_acc, eval_acc])

    plt.savefig(f"../imgs/analyze/testVerify_{dataset}_{threshold_train}_{threshold_eval}_analyze.png")

