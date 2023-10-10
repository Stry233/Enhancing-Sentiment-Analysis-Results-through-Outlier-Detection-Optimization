import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = nn.ReLU()(out)
        return out


# Define the ResNet
class text_ResNet(nn.Module):
    def __init__(self):
        super(text_ResNet, self).__init__()
        self.in_channels = 384  # This might need adjusting depending on the shape of input
        self.rep_dim = 50

        self.conv = nn.Conv1d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Assuming input has 1 channel
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 4 layers
        self.layer1 = self.make_layer(ResidualBlock, 192, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 96, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 68, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.rep_dim, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(self.rep_dim, self.rep_dim)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x.unsqueeze(1))
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class text_ResNet_Autoencoder(nn.Module):
    def __init__(self):
        super(text_ResNet_Autoencoder, self).__init__()
        self.in_channels = 384  # This might need adjusting depending on the shape of input
        self.rep_dim = 50

        self.conv = nn.Conv1d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Assuming input has 1 channel
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 4 layers
        self.layer1 = self.make_layer(ResidualBlock, 192, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 96, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 68, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.rep_dim, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(self.rep_dim, self.rep_dim)

        # Decoder
        self.fc_d = nn.Linear(self.rep_dim, self.rep_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')  # mode is 'linear' for 1D upsampling

        self.conv1 = nn.Conv1d(self.rep_dim, 68, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(68, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.fc_final = nn.Linear(384 * 16, 384)  # to make the final output size match the input

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        if debug: print(f"Input shape: {x.shape}")

        # Encoding
        out = self.conv(x.unsqueeze(1))
        if debug: print(f"After first convolution: {out.shape}")
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        if debug: print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        if debug: print(f"After layer2: {out.shape}")
        out = self.layer3(out)
        if debug: print(f"After layer3: {out.shape}")
        out = self.layer4(out)
        if debug: print(f"After layer4: {out.shape}")

        out = self.avg_pool(out)
        if debug: print(f"After avg_pool: {out.shape}")
        out = out.view(out.size(0), -1)
        if debug: print(f"After flattening: {out.shape}")
        out = self.fc(out)
        if debug: print(f"After FC: {out.shape}")

        x = F.relu(out)
        encoded = x.view(x.size(0), -1)  # flatten the output of the encoder
        # encoded = self.fc(x)
        # if debug: print(f"Encoded shape: {encoded.shape}")

        # Decoding
        x = F.relu(self.fc_d(encoded))
        if debug: print(f"After first decoder FC: {x.shape}")
        x = x.view(x.size(0), 50, 1)
        if debug: print(f"After reshaping: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After first upsample: {x.shape}")

        x = F.relu(self.conv1(x))
        if debug: print(f"After first decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After second upsample: {x.shape}")

        x = F.relu(self.conv2(x))
        if debug: print(f"After second decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After third upsample: {x.shape}")

        x = F.relu(self.conv3(x))
        if debug: print(f"After third decoder conv: {x.shape}")
        x = self.upsample(x)
        if debug: print(f"After fourth upsample: {x.shape}")

        x = torch.sigmoid(self.conv4(x))
        if debug: print(f"Before reshaping and final FC: {x.shape}")

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_final(x)  # final linear layer to match the input size
        if debug: print(f"Final output shape: {x.shape}")

        return x
