import torch
from torch.autograd import Variable
import torch.nn.functional as F

class simpleCNN(torch.nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size= 3, stride= 1, padding= 1)
        self.pool = torch.nn.MaxPool2d(kernel_size= 2, stride= 2, padding= 0)
        self.fc1 = torch.nn.Linear(4608, 64)
        self.fc2 = torch.nn.Linear(64,10)