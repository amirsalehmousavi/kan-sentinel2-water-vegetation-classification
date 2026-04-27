from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, input_size=12):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        '''self.fc5 = nn.Linear(30, 30)
        self.fc6 = nn.Linear(30, 30)
        self.fc7 = nn.Linear(30, 30)'''
        self.fc8 = nn.Linear(30, 20)
        self.fc9 = nn.Linear(20, 3)

        self.dropout1 = nn.Dropout(0.5)  # 50% dropout for 30-node layers
        self.dropout2 = nn.Dropout(0.35)  # 65% keeping probability (30% dropout) for 20-node layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout1(x)

        x = F.relu(self.fc3(x))
        x = self.dropout1(x)

        x = F.relu(self.fc4(x))
        x = self.dropout1(x)

        '''x = F.relu(self.fc5(x))
        x = self.dropout1(x)

        x = F.relu(self.fc6(x))
        x = self.dropout1(x)

        x = F.relu(self.fc7(x))
        x = self.dropout1(x)'''

        x = F.relu(self.fc8(x))
        x = self.dropout2(x)

        x = self.fc9(x)
        x = F.log_softmax(x, dim=1)

        return x
