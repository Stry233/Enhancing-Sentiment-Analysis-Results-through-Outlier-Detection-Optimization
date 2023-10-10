import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseNet


class text_LSTM(BaseNet):
    def __init__(self):
        super(text_LSTM, self).__init__()

        hidden_dim = 256
        self.rep_dim = 50
        # LSTM Layer
        self.lstm = nn.LSTM(384, hidden_dim, batch_first=True)

        # Dense Layer to generate feature of 50 dimensions
        self.hidden2feature = nn.Linear(hidden_dim, self.rep_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)

        # We are using the hidden state of the last LSTM cell as the text feature
        text_feature = self.hidden2feature(lstm_out)

        return text_feature


class text_LSTM_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()
        # encoder
        hidden_dim = 256
        self.rep_dim = 50
        # LSTM Layer
        self.lstm = nn.LSTM(384, hidden_dim, batch_first=True)

        # Dense Layer to generate feature of 50 dimensions
        self.hidden2feature = nn.Linear(hidden_dim, self.rep_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 384)
        )

    def forward(self, inputs):
        # Encoding
        lstm_out, _ = self.lstm(inputs)

        # We are using the hidden state of the last LSTM cell as the text feature
        text_feature = self.hidden2feature(lstm_out)

        # Decoding
        decoded = self.decoder(text_feature)

        return decoded

