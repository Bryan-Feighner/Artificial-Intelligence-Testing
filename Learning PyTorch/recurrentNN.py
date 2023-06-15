import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init

dtype = torch.FloatTensor
inputSize, hiddenSize, outputSize = 7, 6, 1
epochs = 300
sequenceLength = 50
lr = .1
dataTimeSteps = np.linspace(2, 10, sequenceLength + 1)
data = np.sin(dataTimeSteps)
data.resize((sequenceLength + 1, 1))
x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)
weight1 = torch.FloatTensor(inputSize, hiddenSize).type(dtype)
init.normal(weight1, 0.0, 0.4)
weight1 = Variable(weight1, requires_grad = True)
weight2 = torch.FloatTensor(hiddenSize, outputSize).type(dtype)
init.normal(weight2, 0.0, 0.3)
weight2 = Variable(weight2, requires_grad = True)
def forward(input, contextState, weight1, weight2):
    xh = torch.cat((input, contextState), 1)
    contextState = torch.tanh(xh.mm(weight1))
    output = contextState.mm(weight2)
    return (output, contextState)
for i in range(epochs):
    totalLoss = 0
    contextState = Variable(torch.zeros((1, hiddenSize)).type(dtype), requires_grad = True)
    for j in range(x.size(0)):
        input = x[j:(j+1)]
        target = y[j:(j+1)]
        (prediction, contextState) = forward(input, contextState, weight1, weight2)
        loss = (prediction - target).pow(2).sum()/2
        totalLoss += loss
        loss.backward()
        weight1.data -= lr * weight1.grad.data
        weight2.data -= lr * weight2.grad.data
        weight1.grad.data.zero_()
        weight2.grad.data.zero_()
        contextState = Variable(contextState.data)
    if i % 10 == 0:
         print("Epoch: {} loss {}".format(i, totalLoss.item()))

contextState = Variable(torch.zeros((1, hiddenSize)).type(dtype), requires_grad = False)
predictions = []
for i in range(x.size(0)):
    input = x[i:i+1]
    (prediction, contextState) = forward(input, contextState, weight1, weight2)
    contextState = contextState
    predictions.append(prediction.data.numpy().ravel()[0])
pl.scatter(dataTimeSteps[:-1], x.data.numpy(), s = 90, label = "Actual")
pl.scatter(dataTimeSteps[1:], predictions, label = "Predicted")
pl.legend()
pl.show()
