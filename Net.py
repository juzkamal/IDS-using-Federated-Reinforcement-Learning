import torch.nn as nn
import torch.nn.functional as F

from Args import args
n_columns = args.n_columns

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_columns, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, n_columns)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
# Duplicate class that takes n_columns as a parameter during construction
class AnalysisNet(nn.Module):
    def __init__(self, column_count):
        super(AnalysisNet, self).__init__()
        self.column_count = column_count
        
        self.fc1 = nn.Linear(self.column_count, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, self.column_count)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Exports: Net, AnalysisNet