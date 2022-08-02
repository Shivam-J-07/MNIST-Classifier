import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import make_datasets

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
def train_model(net, epochs, train_dataset):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range (epochs):
        for data in train_dataset:
            x, y = data
            net.zero_grad()
            output = net(x.view(-1, 784))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

def test_model(net, test_dataset):
    accurate = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            x, y = data
            output = net(x.view(-1, 784))
            for batch_idx, i in enumerate(output):
                if torch.argmax(i) == y[batch_idx]:
                    accurate+=1
                total+=1
    print("Accuracy: ", round(accurate/total, 3))

net = Net()

BATCH_SIZE = 32

EPOCHS = 3

test_dataset, train_dataset = make_datasets(BATCH_SIZE)

train_model(net, EPOCHS, train_dataset)

test_model(net, test_dataset)