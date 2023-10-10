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


def generate_data(threshold, dataframe, dataset):
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


def start_test(train_data, test_data, model_name="distilbert-base-uncased"):
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
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds,
                                                                   average='weighted')  # 'weighted' for multiclass
        acc = accuracy_score(labels, preds)

        # compute confusion matrix and derive class-specific accuracy
        cm = confusion_matrix(labels, preds)
        class_accuracy = {}
        for i in range(cm.shape[0]):  # loop over each class
            class_accuracy[f'class_{i}_accuracy'] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            **class_accuracy
        }

    # Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=16,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=1e-06,
        save_strategy=IntervalStrategy.NO,  # Added this line to not save weights
        seed=seed,
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
    losses = trainer.state.log_history

    # Print the evaluation results
    for key in sorted(evaluation_results.keys()):
        print(f"{key}: {evaluation_results[key]}")

    # Extract class-specific accuracies from evaluation_results
    class_accuracies = {k: v for k, v in evaluation_results.items() if 'class_' in k}

    return losses, evaluation_results['eval_accuracy'], class_accuracies


def load_data(dataset: str, split: str, percentage: float) -> pd.DataFrame:
    """Loads the specified dataset split."""
    if percentage == 1:
        percentage_prefix = ""
    else:
        percentage_prefix = str(percentage) + "_"
    return pd.read_csv(f"../data/{dataset}/{percentage_prefix}{split}.csv")


def generate_threshold_data(thresholds: list, ground_truth: pd.DataFrame, dataset: str) -> list:
    """Generates data for each threshold."""
    filenames = []
    for threshold in thresholds:
        filename = generate_data(threshold, ground_truth, dataset)
        filenames.append(filename)
    return filenames


def test_thresholds(filenames: list, test_data_path: str, model_name: str) -> tuple:
    """Tests each threshold and returns the logs, accuracies, and class accuracies."""
    logs = []
    accs = []
    class_accuracies = []
    for filename in filenames:
        log, acc, class_accuracy = start_test(filename, test_data_path, model_name=model_name)
        logs.append(log)
        accs.append(acc)
        class_accuracies.append(class_accuracy)
    return logs, accs, class_accuracies


def plot_loss(ax, logs, thresholds, loss_key: str, title: str):
    """Plots the loss trend per epoch for each threshold."""
    losses = [[epoch[loss_key] for epoch in log if loss_key in epoch] for log in logs]
    for i, threshold_losses in enumerate(losses):
        x = np.arange(len(threshold_losses))  # Assuming the x-axis is the epoch number
        coefficients = np.polyfit(x, threshold_losses, 1)
        poly = np.poly1d(coefficients)

        ax.plot(x, poly(x), linestyle='--', label=f'Fit for Threshold {thresholds[i]}')
        ax.plot(threshold_losses, label=f'Threshold {thresholds[i]}', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()


def plot_bars(ax, thresholds, values, title: str, ylabel: str):
    """Plots a bar chart for the given values and labels bars with the value."""
    bars = ax.bar(thresholds, values, width=0.05)
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(value * 100, 5), ha='center', va='bottom')
    ax.set_title(title)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(ylabel)



def plot_class_accuracies(ax, thresholds, class_accuracies):
    """Plots a bar chart for the accuracy of each class for each threshold."""
    class_labels = list(class_accuracies[0].keys())
    n_classes = len(class_labels)
    bar_width = 0.8 / n_classes  # The width of the bars
    opacity = 0.8

    for i, class_label in enumerate(class_labels):
        # Gather the accuracy for this class across all thresholds
        class_accs = [acc[class_label] for acc in class_accuracies]

        # Compute the position for the bar for this class
        bar_positions = [j + i * bar_width for j in range(len(thresholds))]

        # Plot the bar for this class
        ax.bar(bar_positions, class_accs, bar_width, alpha=opacity, label=class_label)

        # Add numbers to each bar
        for pos, yval in zip(bar_positions, class_accs):
            ax.text(pos, yval, round(yval * 100, 5), ha='center', va='bottom')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Class Accuracies per Threshold')
    ax.set_xticks(np.arange(len(thresholds)) + bar_width / 2)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.legend()


if __name__ == "__main__":
    dataset = "mteb_reviews"
    model_name = "distilbert-base-uncased"
    portion = 1
    thresholds = [0.2, 0.4, 0.6, 0.8, 1]

    # Load data
    ground_truth_train = load_data(dataset, "train", portion)  # e.g. portion = 0.01 ==> 0.01_train.csv for 1% twitter
    ground_truth_test = load_data(dataset, "test", portion)

    # Generate data for each threshold
    filenames = generate_threshold_data(thresholds, ground_truth_train, dataset)

    # Test each threshold
    if portion == 1:
        percentage_prefix = ""
    else:
        percentage_prefix = str(portion) + "_"
    test_file_location = f"../data/{dataset}/{percentage_prefix}test.csv"
    logs, accs, class_accuracies = test_thresholds(filenames, test_file_location, model_name="distilbert-base-uncased")

    # Normalize the accuracies to [0, 1]
    norm_accs = [(x - min(accs)) / (max(accs) - min(accs)) for x in accs]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(5, 1, figsize=(10, 22))

    # Plot train and evaluation losses
    plot_loss(ax[0], logs, thresholds, 'loss', 'Train Loss trend per epoch')
    plot_loss(ax[1], logs, thresholds, 'eval_loss', 'Evaluation Loss trend per epoch')

    # Plot normalized and absolute accuracies
    plot_bars(ax[2], thresholds, norm_accs, 'Normed accuracies', 'Normalized Accuracy')
    plot_bars(ax[3], thresholds, accs, 'Absolute accuracies', 'Accuracy')

    # multi class acc
    plot_class_accuracies(ax[4], thresholds, class_accuracies)

    # Display the figure
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../imgs/analyze/paper_fig/{dataset}_{model_name}_analyze.png")

