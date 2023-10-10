import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_data(threshold, dataframe):
    file_path_class0 = os.path.join('..', 'score', dataset, 'class0', 'mask_class0.json')
    file_path_class1 = os.path.join('..', 'score', dataset, 'class1', 'mask_class1.json')

    loaded_predicted_class0 = load_json(file_path_class0)
    loaded_predicted_class1 = load_json(file_path_class1)

    mask_class0 = [score[0] for score in loaded_predicted_class0 if score[2] <= threshold]
    mask_class1 = [score[0] for score in loaded_predicted_class1 if score[2] <= threshold]

    # Merge the masks and remove duplicates
    mask = list(set(mask_class0 + mask_class1))

    # Filter the DataFrame based on the index list
    filtered_df = dataframe.loc[mask].dropna()

    # Write the filtered DataFrame to a new CSV file
    data_dir = os.path.join("..", "data", dataset, f"{threshold}_train.csv")
    filtered_df.to_csv(data_dir, index=False)

    return data_dir


def start_test(train_data, test_data, model_name="LogisticRegression"):
    # Load the data from CSV file
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    # Extract features and labels
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    # Convert text data to TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Specify the model
    if model_name == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=42)
    elif model_name == "LDA":
        model = LinearDiscriminantAnalysis()
        # Convert sparse matrices to dense
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred,
                                                               average='weighted', labels=np.unique(y_pred))
    acc = accuracy_score(y_test, y_pred)

    evaluation_results = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    # Print the evaluation results
    for key in sorted(evaluation_results.keys()):
        print(f"{key}: {evaluation_results[key]}")

    return evaluation_results


if __name__ == "__main__":
    dataset = "mteb_imdb"
    model_name = "LDA"

    ground_truth_train = pd.read_csv(f"../data/{dataset}/train.csv")  # 0.01_train.csv for 1% twitter
    ground_truth_test = pd.read_csv(f"../data/{dataset}/test.csv")
    thresholds = [0.2, 0.4, 0.6, 0.8, 1]
    filenames = []
    for threshold in thresholds:
        filename = generate_data(threshold, ground_truth_train)
        filenames.append(filename)

    logs = []
    accs = []
    for filename in filenames:
        acc = start_test(filename, f"../data/{dataset}/test.csv", model_name=model_name)['accuracy']
        accs.append(acc)

    # visualize
    fig, axs = plt.subplots(2, 1, figsize=(10, 20))

    # Plot actual accuracies
    bars = axs[0].bar(thresholds, accs, width=0.05)
    axs[0].set_title('Final accuracies')
    # axs[0].set_xlabel('Threshold')
    axs[0].set_ylabel('Actual Accuracy')
    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, round(yval*100, 5), ha='center', va='bottom')

    # Normalize the accuracies to [0, 1]
    max_acc = max(accs)
    min_acc = min(accs)
    norm_accs = [(x - min_acc) / (max_acc - min_acc) for x in accs]

    # Plot normalized accuracies
    bars = axs[1].bar(thresholds, norm_accs, width=0.05)
    axs[1].set_title('Final accuracies (Normalized)')
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylabel('Normalized Accuracy')
    for bar in bars:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, round(yval*100, 5), ha='center', va='bottom')

    # Adjust the spacing between plots
    plt.subplots_adjust(hspace = 0.3)

    # Display the figure
    plt.tight_layout()
    plt.savefig(f"../imgs/analyze/ML_{model_name}_{dataset}.jpg")
    plt.show()


