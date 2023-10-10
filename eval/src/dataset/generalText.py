import pandas as pd
import torch
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM

from base import base_dataset
from dataset.preprocessing import get_target_label_idx


class GeneralTextDataset(base_dataset.BaseADDataset):

    def __init__(self, root, normal_class, need_tranform):
        super().__init__(root=root)

        self.normal_classes = tuple([normal_class])

        # Load csv file
        self.train_df = pd.read_csv(root + "train.csv")
        self.test_df = pd.read_csv(root + "test.csv")

        # outlier classes defined
        self.n_classes = self.train_df['label'].nunique()
        self.outlier_classes = list(range(0, self.n_classes))

        self.outlier_classes.remove(normal_class)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create dataset
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes)) if need_tranform else None
        self.train_set = self.create_dataset(self.train_df, method='sentence-embedding')
        self.test_set = self.create_dataset(self.test_df, method='sentence-embedding',
                                            target_transform=target_transform)

        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(self.train_set.targets, self.normal_classes)
        self.train_set = Subset(self.train_set, train_idx_normal)

        if not need_tranform:  # case: need evaluate in infer
            test_idx_normal = get_target_label_idx(self.test_set.targets, self.normal_classes)
            self.test_set = Subset(self.test_set, test_idx_normal)

    def create_dataset(self, df, method='none', target_transform=None):
        # Convert texts and labels into tensors
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        if method == 'word-embedding':
            # Tokenize the texts and convert them to embeddings
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                inputs = self.model(**inputs).last_hidden_state
        elif method == 'sentence-embedding':
            # Convert sentences to embeddings
            with torch.no_grad():
                inputs = self.sentence_embedding_model.encode(texts, convert_to_tensor=True)
        elif method == 'none':
            # Convert sentences to embeddings
            with torch.no_grad():
                inputs = texts
        else:
            raise ValueError('Invalid method: choose either "tokenizer", "word-embedding" or "sentence-embedding"')

        # Return a dictionary with inputs and labels
        dataset = MyTextDataset({'inputs': inputs, 'labels': labels}, target_transform=target_transform)
        return dataset

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) \
            -> (DataLoader, DataLoader):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader

    def __getitem__(self, index):
        inputs = {key: val[index] for key, val in self.train_set['inputs'].items()}
        label = self.train_set['labels'][index]
        return inputs, label, index

    def __len__(self):
        return len(self.train_set['inputs']['input_ids'])


class MyTextDataset(Dataset):
    def __init__(self, data, target_transform):
        self.data = data['inputs']
        self.targets = data['labels']
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets, idx
