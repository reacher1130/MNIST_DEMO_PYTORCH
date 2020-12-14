from torch import nn
from dataloader import *
from train import *
from model import Rnn_Classify, AlexNet, AlexNet_small, ResNet
import matplotlib.pyplot as plt
from thop import profile
if __name__ == '__main__':

    lr, batch_size, num_epochs, momentum = 0.01, 64, 10, 0.9
    loss_fun = 'class'
    optimizer_fun = 'SGD'
    save_model = True

    net = AlexNet()

    if loss_fun == 'reg':
        criterion = nn.MSELoss()
    elif loss_fun == 'class':
        criterion = nn.CrossEntropyLoss()
    if optimizer_fun == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum)
    elif optimizer_fun == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_data_mnist(batch_size)

    train_on_epoch(net, train_data, test_data, num_epochs, optimizer, criterion, device)
    print(net)
    # 加载模型进行测试
    state_dict = torch.load('mnist_cnn.pth', map_location=device)
    net.load_state_dict(state_dict)
    print('loading trained model finished')
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    # with torch.no_grad():
    #     print_mnist(test_data)
    #     data_iter = iter(test_data)
    #     images, labels = data_iter.next()
    #     print('GroundTruth: ', ' '.join('%d' % labels[j] for j in range(64)))
    #     test_out = net(images)
    #
    #     _, predicted = torch.max(test_out, dim=1)
    #     print('Predicted: ', ' '.join('%d' % predicted[j]
    #                                   for j in range(64)))

    # predict_ten_classes(net, test_data, device)



