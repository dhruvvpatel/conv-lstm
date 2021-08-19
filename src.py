import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('root', help='Root folder to download MNIST dataset at')
args = parser.parse_args()
root = args.root

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# hyper-params
# --- general ---
EPOCHS = 3
lr_rate = 0.001
BS_TRAIN = 64
BS_TEST = 1000

# --- rnn ---
# feature vector length o/p from cnn : 1024
seq_length = 32
input_size = 32
hidden_size = 128
num_layers = 2
num_classes = 10


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=root,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=root,
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BS_TRAIN,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BS_TEST,
                                          shuffle=False)


# CNN
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, 5, padding=2)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*28*28, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.drop(self.conv2(x)))
        x = x.view(-1, 20*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 1024)

        return x


# RNN ( many-to-one )
class RNN(nn.Module):

    def __init__(self, i_size, h_size, n_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layers

        self.lstm = nn.LSTM(i_size, h_size, n_layers, batch_first=True)
        self.fc = nn.Linear(h_size, num_classes)


    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


# ---------------------------
#   ULTIMATE CNN+RNN NETWORK
# ---------------------------
class NETWORK(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, seq_len):
        super(NETWORK, self).__init__()

        self.cnn = CNN()
        self.rnn = RNN(input_size, hidden_size, num_layers, num_classes)

    def forward(self, x):
        f_vector = self.cnn(x)
        x = f_vector.view(-1, seq_length, input_size)
        out = self.rnn(x)

        return out

# ------------------------------------------
torch.manual_seed(44)
net = NETWORK(input_size, hidden_size, num_layers, num_classes, seq_length).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate)

# train the model
total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        output = net(images)
        loss = criterion(output, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))



# test the model
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

