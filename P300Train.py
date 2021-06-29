from math import floor

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Device configuration
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn import preprocessing
import torch.optim as optim

from utils import conv1d_output_shape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 35  # change this base on the length of x
hidden_size = 64
num_classes = 2
num_epochs = 100
batch_size = 128
learning_rate = 0.001

x = np.load('P300_062221-062521/data.npy')
y = np.load('P300_062221-062521/labels.npy')
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# stratified split for balancing classes before and after train-test-split
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.1, random_state=42, stratify=y)

dl_train = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)))
dl_test = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)))

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_filter=8, kernel_size=5):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(3, out_channels=num_filter, kernel_size=kernel_size, stride=1)
        self.pool = nn.MaxPool1d(2)

        fc_input_shape = conv1d_output_shape(input_size, kernel_size=kernel_size, stride=1)
        fc_input_shape = floor(fc_input_shape/2) * num_filter

        self.fc1 = nn.Linear(fc_input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    train_loss = 0.0
    for i, data in enumerate(dl_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        running_loss += loss.item()
        if i % 3 == 0:    # print every 3 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        # test validation every epoch
        test_correct = 0
        test_total = 0
        train_correct = 0
        train_total = 0
        val_loss = 0.
        trian_loss_e = 0.
        with torch.no_grad():
            for data in dl_test:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += np.sum(predicted == labels).item()
                val_loss += criterion(outputs, labels)
            for data in dl_train:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += np.sum(predicted == labels).item()
                trian_loss_e += criterion(outputs, labels)

        print('Epoch {0}: train loss: {1}, train acc {2}'
              'val loss {3}, val acc {4}'.format(epoch, train_loss, train_correct/train_total, val_loss, test_correct/test_total))

print('Finished Training')