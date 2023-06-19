import torch
import torch.nn as nn
import random

inputLayer, hiddenLayer, outputLayer, batchSize = 1, 10, 1, 1
x = torch.randn(batchSize, inputLayer)
y = torch.tensor([[random.random()*10]])
model = nn.Sequential(nn.Linear(inputLayer, hiddenLayer), nn.ReLU(), nn.Linear(hiddenLayer, outputLayer), nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = .05)
for time in range(1000):
    prediction = model(x)
    loss = criterion(prediction, y)
    print('Time: ', time, ' x: ', x, ' y: ', y, 'prediction: ', prediction, ' loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
