import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.numGames = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11,1024,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
            
    def getState(self, game):
        head = game.snake[0]
        pointL = Point(head.x - 20, head.y)
        pointR = Point(head.x + 20, head.y)
        pointU = Point(head.x, head.y - 20)
        pointD = Point(head.x, head.y + 20)

        dirL = game.direction == Direction.LEFT
        dirR = game.direction == Direction.RIGHT
        dirU = game.direction == Direction.UP
        dirD = game.direction == Direction.DOWN

        state = [
            (dirR and game.is_collision(pointR)) or (dirL and game.is_collision(pointL)) or (dirU and game.is_collision(pointU)) or (dirD and game.is_collision(pointD)),
            (dirU and game.is_collision(pointU)) or (dirD and game.is_collision(pointD)) or (dirL and game.is_collision(pointL)) or (dirR and game.is_collision(pointR)),
            (dirD and game.is_collision(pointD)) or (dirU and game.is_collision(pointU)) or (dirR and game.is_collision(pointR)) or (dirL and game.is_collision(pointL)),
            dirL,
            dirR,
            dirU,
            dirD,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype = int)
    
    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver))
    
    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            miniSample = self.memory
        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)
    
    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)
    
    def getAction(self, state):
        self.epsilon = 80 - self.numGames
        finalMove = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1
        return finalMove

def train():
    plotScore = []
    plotMeanScores = []
    totalScore = 0
    meanScore = 0.0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        stateOld = agent.getState(game)
        finalMove = agent.getAction(stateOld)
        reward, gameOver, score = game.play_step(finalMove)
        stateNew = agent.getState(game)
        agent.trainShortMemory(stateOld, finalMove, reward, stateNew, gameOver)
        agent.remember(stateOld, finalMove, reward, stateNew, gameOver)
        if gameOver:
            game.reset()
            agent.numGames += 1
            agent.trainLongMemory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.numGames, 'Score', score, 'Record', record)
            plotScore.append(score)
            totalScore =+ score
            meanScore = totalScore / agent.numGames
            plotMeanScores.append(meanScore)
            plot(plotScore, plotMeanScores)

    
if __name__ == '__main__':
    train()