import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class SAVEE_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        # 数据有time-dependency: CNN不好用
        #  --》TIM-NET
        #  --》再试一个 text dataset: common-sense:
        #       --> does higher acc applied in the original text data? (测试集不可以洗！)

        #  --> ASR: 大部分都是非监督学习：可不可以用这个获得一些好data pts然后拿这个做一些监督学习 --> 更快更好的学习方法
        #  --> validation set做一边Deep SVDD, check outlier point 是否是model做错的那一部分


        # 优先：
        #  --> test中错（注意test没有清洗过）的那些和deep svdd的结果的放在一起：如果错的部分很大一部分是outlier
        #       --》 1. 结论：classifier在实际使用中，可以先用deep SVDD check其是否为outlier（作为一个filter） 然后再用模型进行reason
        #
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)  # 1 input channel
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(15872, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = x.view(-1, 1, 254, 39)  # Reshape the data to have 1 channel
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = F.leaky_relu(x)
        return x


class SAVEE_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)  # 1 input channel
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(15872, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, 15872, bias=False)
        self.bn1d2 = nn.BatchNorm1d(15872, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 5, bias=False, padding=2)  # Output has 1 channel
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, debug=False):
        x = x.view(-1, 1, 254, 39)  # Reshape the data to have 1 channel
        if debug: print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        if debug: print(f"After conv1: {x.shape}")
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        if debug: print(f"After pool1: {x.shape}")
        x = self.conv2(x)
        if debug: print(f"After conv2: {x.shape}")
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        if debug: print(f"After pool2: {x.shape}")
        x = self.conv3(x)
        if debug: print(f"After conv3: {x.shape}")
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        if debug: print(f"After pool3: {x.shape}")
        x = x.view(x.size(0), -1)
        if debug: print(f"After flattening: {x.shape}")
        x = self.bn1d(self.fc1(x))
        if debug: print(f"After fc1 and bn1d: {x.shape}")
        x = F.leaky_relu(x)

        # Decoder
        if debug: print(f"Start of decoder: {x.shape}")
        x = self.fc2(x)
        if debug: print(f"After fc2: {x.shape}")
        x = self.bn1d2(x)
        if debug: print(f"After bn1d2: {x.shape}")
        x = x.view(x.size(0), 128, 31, 4)  # Reshape to match the shape before the flattening
        if debug: print(f"After reshaping: {x.shape}")
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(x, size=(63, 9))  # upsample to match the size after conv2
        if debug: print(f"After deconv1: {x.shape}")
        x = F.leaky_relu(self.bn2d4(x))
        x = self.deconv2(x)
        x = F.interpolate(x, size=(127, 19))  # upsample to match the size after conv1
        if debug: print(f"After deconv2: {x.shape}")
        x = F.leaky_relu(self.bn2d5(x))
        x = self.deconv3(x)
        x = F.interpolate(x, size=(254, 39))  # upsample to match the original input size
        if debug: print(f"After deconv3: {x.shape}")
        x = torch.sigmoid(x)  # apply a sigmoid to ensure the output is between 0 and 1
        if debug: print(f"Final output shape: {x.shape}")

        return x



