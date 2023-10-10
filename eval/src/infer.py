import json
import logging
import os

import torch

from dataset.main import load_dataset
from deepSVDD import DeepSVDD


def calculate_label_score(data, deepSVDD):
    """
    Calculate labels and scores for given data.

    Parameters:
    data (Tuple[torch.Tensor]): Tuple of inputs, labels and indices from the DataLoader.
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Ground truth labels.
        idx (torch.Tensor): Indices of the data.
    deepSVDD (DeepSVDD): The neural network model to use for prediction.

    Returns:
    List[Tuple[int, int, float]]: List of tuples with indices, labels and calculated scores.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    R = torch.tensor(deepSVDD.R, device=device)  # radius R initialized with 0 by default.
    c = torch.tensor(deepSVDD.c, device=device) if deepSVDD.c is not None else None
    nu = deepSVDD.nu

    inputs, labels, idx = data
    inputs = inputs.to(device)
    net = deepSVDD.net.to(device)
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)

    if deepSVDD.objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist

    return [idx.cpu().data.numpy().tolist()[0],
            labels.cpu().data.numpy().tolist()[0],
            scores.cpu().data.numpy().tolist()[0]]


def infer(dataset_name='mteb_counterfactual', net_name='text_LSTM', data_path='../data/',
          save_path='../score/mteb_counterfactual/', eval_train=True, mask=None, normal_class=0):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not eval_train:
        save_path = save_path[:-1] + "_eval" + "/"
    dir_path = save_path + f'class{normal_class}/'
    log_file = dir_path + 'log.txt'
    # Check if the directory exists, if not, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load data
    data_path = data_path + dataset_name + "/"
    dataset = load_dataset(dataset_name, data_path, normal_class, need_tranform=False)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD('one-class', 0.1)
    deep_SVDD.set_network(net_name)

    # Load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    deep_SVDD.load_model(model_path=f"../log/{dataset_name}/class{normal_class}/model.tar", load_ae=False)

    # Get train data loader
    train_loader, test_loader = dataset.loaders(batch_size=1, shuffle_train=False, shuffle_test=False)

    if eval_train:
        eval_loader = train_loader
    else:
        eval_loader = test_loader

    logger.info('Start evaluation...')
    all_res = []
    with torch.no_grad():
        for data in eval_loader:
            if mask is None or data[2] not in mask:
                content = calculate_label_score(data, deep_SVDD)
                all_res.append(content)  # idx, label, scores

    # Extract scores from all_res
    all_scores = [item[2] for item in all_res]

    # Normalize scores
    min_score = min(all_scores)
    max_score = max(all_scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in all_scores]

    # Replace original scores with normalized scores
    for i in range(len(all_res)):
        all_res[i][2] = normalized_scores[i]

    # Convert probabilities to binary predictions with 0.5 as threshold
    # predicted = [1 if prob >= 0.5 else 0 for prob in normal_score]
    with open(save_path+f'class{normal_class}/mask_class{normal_class}.json', 'w') as f:
        json.dump(all_res, f)


if __name__ == '__main__':
    dataset_name = "C4_sentiment-analysis"
    number_of_classes = 3
    eval_train = True  # infer on train set?
    save_path = f'../score/{dataset_name}/'
    for normal_class in range(number_of_classes):
        infer(dataset_name=dataset_name, save_path=save_path, normal_class=normal_class, eval_train=eval_train)
