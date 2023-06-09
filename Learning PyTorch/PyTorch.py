# Import both required libraries
import torch
import torch.nn as nn
# Set sizes for all layers and batch
inputLayer, hiddenLayer, outputLayer, batchSize = 10, 5, 1, 10
# Create input data with a nomal distribution using torch.randn(yInputSize, xInputSize)
x = torch.randn(batchSize, inputLayer)
# Create a tensor with 10 values
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
# Create squential model
model = nn.Sequential(nn.Linear(inputLayer, hiddenLayer), nn.ReLU(), nn.Linear(hiddenLayer, outputLayer), nn.Sigmoid())
# Create loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# Run model 100 times
for time in range (100):
    # Perform first prediction
    yPrediction = model(x)
    # Calculate loss and print to terminal
    loss = criterion(yPrediction, y)
    print('time: ', time,' loss: ', loss.item())
    # Zero gradient, update weights in optimizer
    optimizer.zero_grad()
    # Perform backward pass
    loss.backward()
    # Update parameters run again
    optimizer.step()
