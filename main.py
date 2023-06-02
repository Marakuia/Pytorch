import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import torchvision.models as models


class Network(nn.Module):
    def __init__(self, _num_layers, _conv_parameters, _activation_funct, _model_resnet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = _num_layers
        self.activation_funct = _activation_funct
        self.num_filter, self.size_filter, self.stride, self.padding = _conv_parameters
        self.model_resnet = _model_resnet
        act_func = None

        match self.activation_funct:
            case "relu":
                act_func = nn.ReLU()
            case "sigmoid":
                act_func = nn.Sigmoid()
            case "tanh":
                act_func = nn.Tanh()
            case "softmax":
                act_func = nn.Softmax()

        layers_list = []

        # input_layer1 = nn.Linear(self.model_resnet.fc.in_features, self.num_filter)
        input_layer = nn.Conv2d(128, self.num_filter, self.size_filter, self.stride, self.padding)

        conv_layers = nn.Conv2d(self.num_filter, self.num_filter, self.size_filter, self.stride, self.padding)
        output_layer = nn.Linear(self.num_filter, 10)


        # layers_list.append(input_layer1)

        layers_list.append(input_layer)
        layers_list.append(act_func)
        layers_list.append(nn.MaxPool2d(2, 1))

        for i in range(self.num_layers):
            layers_list.append(conv_layers)
            layers_list.append(act_func)
            layers_list.append(nn.MaxPool2d(2, 1))

        layers_list.append(nn.Flatten())
        layers_list.append(nn.Linear(1024, self.num_filter))
        layers_list.append(act_func)
        layers_list.append(output_layer)
        # layers_list.append(act_func)

        # input_layer = nn.Linear(self.model_resnet.fc.in_features, int(self.num_filter / 2))
        # av_layer = nn.Linear(input_layer.out_features, self.num_filter)
        # av_layer1 = nn.Linear(self.num_filter, self.num_filter)
        #
        # output_layer = nn.Linear(self.num_filter, 10)
        #
        # layers_list.append(input_layer)
        # layers_list.append(act_func)
        # layers_list.append(av_layer)
        # layers_list.append(act_func)
        # layers_list.append(av_layer1)
        # layers_list.append(act_func)
        # layers_list.append(nn.Dropout())
        # layers_list.append(output_layer)


        layers = nn.Sequential(*layers_list)
        self.model_resnet.fc = layers

    def forward(self, x):
        # x = self.model_resnet(x)
        # return x
        x = self.model_resnet.conv1(x)
        x = self.model_resnet.bn1(x)
        x = self.model_resnet.relu(x)
        x = self.model_resnet.maxpool(x)

        x = self.model_resnet.layer1(x)
        x = self.model_resnet.layer2(x)
        x = self.model_resnet.layer3(x)
        x = self.model_resnet.layer4(x)

        x = self.model_resnet.avgpool(x)
        x = self.model_resnet.fc(x.reshape(x.size(0), 128, 8, -1))


        return x


def train(model, optimiz, loss_fn, train_loader, validation_loader, train_device, epochs=100):
    train_losses = []
    valid_losses = []
    train_accuracy = []
    val_accuracy = []
    model.to(train_device)
    print(train_device)
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        model.train()
        for image, image_class in train_loader:
            image, image_class = image.to(train_device), image_class.to(train_device)
            print(image.shape)
            optimiz.zero_grad()
            predict = model(image)
            print(predict.shape)
            loss = loss_fn(predict, image_class)
            loss.backward()
            optimiz.step()
            train_loss += loss.item() * image.size(0)
            train_acc += accuracy(predict, image_class)

        model.eval()
        for val_image, val_image_class in validation_loader:
            val_image, val_image_class = val_image.to(train_device), val_image_class.to(train_device)
            predict = model(val_image)
            loss = loss_fn(predict, val_image_class)
            valid_loss += loss.item() * val_image.size(0)
            val_acc += accuracy(predict, val_image_class)

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(validation_loader)
        train_acc = train_acc / len(train_loader)
        val_acc = val_acc / len(validation_loader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

        print('Epoch:{} Train accuracy:{:.4f} Validation accuracy:{:.4f}'.format(epoch, train_acc, val_acc))
        if train_acc - val_acc > 0.15:
            print("\nWith further iterations, overfitting is possible \nBreak loop, epoch: {}".format(epoch))
            break

    return train_losses, valid_losses, train_accuracy, val_accuracy


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    # print("Out: {}\nPredict: {}\nLabels: {}".format(outputs, preds, labels))
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


NUMBER_OF_FILTERS = 256
SIZE_OF_FILTERS = 3
STRIDE = 1
PADDING = 1
LEARNING_RATE = 0.001

resnet_model = models.resnet50(pretrained=True)

resnet_model.requires_grad = False

# resnet_model.fc.requires_grad = True

print(resnet_model.fc.in_features)
net = Network(2, (NUMBER_OF_FILTERS, SIZE_OF_FILTERS, STRIDE, PADDING), "relu", resnet_model)
print(net)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
                              ])

train_data = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
validation_data = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)
validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=2)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

resnet_model.requires_grad = False

all_train_losses, validation_losses, all_train_accuracy, validation_accuracy = \
    train(net, optimizer, loss_function, train_data_loader, validation_data_loader, device)


plt.subplot(2, 1, 1)
plt.plot(np.arange(len(all_train_losses)), np.array(all_train_losses), label="Train losses")
plt.plot(np.arange(len(validation_losses)), np.array(validation_losses), label="Validation losses")
plt.legend()
plt.title("Cross Entropy Loss")
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(all_train_accuracy)), np.array(all_train_accuracy), label="Train accuracy")
plt.plot(np.arange(len(validation_accuracy)), np.array(validation_accuracy), label="Validation accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

torch.save(net.state_dict(), 'cifar10-cnn.pth')
