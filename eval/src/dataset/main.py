from .generalText import GeneralTextDataset
from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset

def load_dataset(dataset_name, data_path, normal_class, need_tranform=True):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'twitter',
                            'mteb_counterfactual', 'mteb_imdb', 'mteb_toxic', 'mteb_tweet_sentiment', 'mteb_emotion',
                            'hate_speech18', 'sst2', 'yelp_polarity', 'mteb_reviews', 'C4_sentiment-analysis')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name in ['twitter', 'mteb_counterfactual', 'mteb_imdb', 'mteb_toxic',
                        'mteb_tweet_sentiment', 'mteb_emotion', 'hate_speech18', 'sst2',
                        'yelp_polarity', 'mteb_reviews', 'C4_sentiment-analysis']:
        dataset = GeneralTextDataset(root=data_path, normal_class=normal_class, need_tranform=need_tranform)

    return dataset
