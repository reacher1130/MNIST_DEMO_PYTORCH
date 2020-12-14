from abc import ABC
import torch.nn as nn
import torch.nn.functional as F


class Rnn_Classify(nn.Module, ABC):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(Rnn_Classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # 使用两层lstm
        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个的rnn使用全连接的到最后的输出结果

    def forward(self, x):
        # x的大小为（batch，1，28,28），所以我们需要将其转化为rnn的输入格式（28，batch，28）
        x = x.squeeze()  # 去掉（batch，1,28,28）中的1，变成（batch， 28,28）
        x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成（batch，28,28）
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的out是（28， batch， hidden_feature）
        out = out[-1, :, :]  # 取序列中的最后一个，大小是（batch， hidden_feature)
        out = self.classifier(out)  # 得到分类结果
        return out


class AlexNet(nn.Module, ABC):
    '''
    类似AlexNet的神经网络，因为电脑配置及MNIST数据集图片尺寸问题，将Kernel_size和stride都改小了
    '''
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNet_small(nn.Module, ABC):

    def __init__(self, num_classes=10):
        super(AlexNet_small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x


class Basicblock(nn.Module, ABC):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module, ABC):
    def __init__(self, block=Basicblock, num_block=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)  #16 32 64
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        self.block4 = self._make_layer(block, 128, num_block[3], stride=2)

        self.out_layer = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.block1(x)  # [200, 64, 28, 28]
        x = self.block2(x)  # [200, 128, 14, 14]
        x = self.block3(x)  # [200, 256, 7, 7]
        x = self.block4(x)
        # out = self.block4(out)
        x = F.avg_pool2d(x, 4)  # [200, 256, 1, 1]   7
        x = x.view(x.size(0), -1)  # [200,256]
        out = self.out_layer(x)
        return out
