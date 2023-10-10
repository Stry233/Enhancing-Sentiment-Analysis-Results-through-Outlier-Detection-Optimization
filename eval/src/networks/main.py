from networks.minist.mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from networks.cifar10.cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from networks.cifar10.cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from networks.cifar10.cifar10_ResNet import CIFAR10_ResNet, CIFAR10_ResNet_Autoencoder
from networks.ASR.savee_TIMNet import SAVEE_TIMNET
from networks.text.text_LSTM import text_LSTM_Autoencoder, text_LSTM

from networks.text.text_ResNet import text_ResNet, text_ResNet_Autoencoder
from networks.ASR.savee_LetNet import SAVEE_LeNet_Autoencoder, SAVEE_LeNet
def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'cifar10_ResNet', 'text_Transformer',
                            'text_ResNet', 'savee_LeNet', 'text_LSTM', 'savee_TIMNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'cifar10_ResNet':
        net = CIFAR10_ResNet()

    if net_name == 'text_ResNet':
        net = text_ResNet()

    if net_name == 'text_LSTM':
        net = text_LSTM()

    if net_name == 'savee_LeNet':
        net = SAVEE_LeNet()

    if net_name == 'savee_TIMNet':
        net = SAVEE_TIMNET()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'cifar10_ResNet', 'text_Transformer',
                            'text_ResNet', 'savee_LeNet', 'text_LSTM', 'savee_TIMNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'cifar10_ResNet':
        ae_net = CIFAR10_ResNet_Autoencoder()

    if net_name == 'text_ResNet':
        ae_net = text_ResNet_Autoencoder()

    if net_name == 'text_LSTM':
        ae_net = text_LSTM_Autoencoder()

    if net_name == 'savee_LeNet':
        ae_net = SAVEE_LeNet_Autoencoder()

    if net_name == 'savee_TIMNet':
        ae_net = SAVEE_LeNet_Autoencoder()

    return ae_net
