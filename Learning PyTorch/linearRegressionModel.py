import torch
import torch.nn as nn
from torch.autograd import Variable
xTrain = x.reshape(-1,1).asType('float32')
xTrain = y.reshape(-1,1).asTypw('float32')

class linearRegressionModel(nn.Module):
    def __init__(self, inputDimension, outputDimension):
        super(linearRegressionModel, self).__init__()
        self.linear = nn.Linear(inputDimension, outputDimension)