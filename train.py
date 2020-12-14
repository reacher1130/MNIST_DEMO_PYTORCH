from tensorboardX import SummaryWriter
import torch
import numpy as np
import datetime
from matplotlib import pyplot as plt


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def predict_ten_classes(test_net, test_loader, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            test_net = test_net.to(device)
            outputs = test_net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %d : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))


def plot_figure(epoch, train_acc, train_loss, test_acc, test_loss):
    x = range(0, epoch)
    plt.subplot(2, 1, 1)
    plt.plot(x, train_acc, 'bo-', label='Training accuracy')
    plt.plot(x, test_acc, 'r*-', label='validation accuracy')
    plt.title('Training and validation accuracy on AlexNet')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy: %')
    b = max(train_acc)
    a = train_acc.index(b)
    plt.annotate('%s' % round(b, 4), xy=(a, b), xytext=(-20, 10), textcoords='offset points', color='b')
    d = test_acc[a]
    plt.annotate('%s' % round(d, 4), xy=(a, d), xytext=(-20, -10), textcoords='offset points', color='r')

    plt.subplot(2, 1, 2)
    plt.plot(x, train_loss, 'bo-', label='Training loss')
    plt.plot(x, test_loss, 'r*-', label='validation loss')
    plt.title('Training and validation loss on AlexNet')
    plt.legend()
    plt.ylabel('Loss:')
    e = min(train_loss)
    f = train_loss.index(e)
    plt.annotate('%s' % round(e, 4), xy=(f, e), xytext=(-20, 10), textcoords='offset points', color='b')
    g = test_loss[f]
    plt.annotate('%s' % round(g, 4), xy=(f, g), xytext=(-20, -10), textcoords='offset points', color='r')

    plt.savefig('accuracy_loss.jpg', dpi=300)
    plt.show()


# 可以传入writer
def train_on_epoch(model, train_data, test_data, num_epochs, optimizer, criterion, device):
    train_Loss_list = []
    test_Loss_list = []
    train_Accuracy_list = []
    test_Accuracy_list = []
    model = model.to(device)
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        model = model.train()
        for im, label in train_data:
            im = im.to(device)
            label = label.to(device)
            # forward
            output = model(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        # test
        if test_data is not None:
            valid_loss = 0
            valid_acc = 0
            model = model.eval()
            for im, label in test_data:
                im = im.to(device)
                label = label.to(device)
                output = model(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(test_data),
                       valid_acc / len(test_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        # 将训练测试数据存储下来
        train_Loss_list.append(train_loss / len(train_data))
        train_Accuracy_list.append(train_acc / len(train_data))
        test_Loss_list.append(valid_loss / len(test_data))
        test_Accuracy_list.append(valid_acc / len(test_data))

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print('saved model!')
    plot_figure(num_epochs, train_Accuracy_list, train_Loss_list, test_Accuracy_list, test_Loss_list)
