#import the needed modules
import numpy as np
import pandas as pd
import matplotlib as plt
import gym
import os
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Environment we will train our agent to play
env = gym.make("CartPole-v1")
#Store the size of our space environment
stateSize = env.observation_space.shape[0]
#Store the number of possible actions
actionSize = env.action_space.n


batchSize = 32
#Number of episodes to train (number of times the agent will repeat playing the game)
numOfEpisodes = 2500
#The max number of steps the agent will take in each spisodes
numOfSteps = 2000


#Directory to store the trained agent
outputDir = 'output/cartpole'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


class Agent:

    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize

        # set hyperparameters

        # Save the last 200 steps
        self.memory = deque(maxlen=2000)

        # The weight will give for the future rewards if
        # gamma close to 0 the agent will tend to consider only immediate rewards
        # else if gamma close to 1 the agent will give future rewards more weights
        self.gamma = 0.96
        self.epsilon = 1.0  # Exploration rate
        self.epsilonDecay = 0.995
        self.epsilonMin = 0.01 # Stope decreasing epsilon when it reachs this value
        self.learningRate = 0.001
        self.model = self._build_model()

    # Buidling our neural network
    def _build_model(self):
        model = Sequential()

        model.add(Dense(64, input_dim=self.stateSize, activation='elu'))
        model.add(Dense(64, activation='elu'))
        model.add(Dense(32, activation='elu'))
        model.add(Dense(16, activation='elu'))
        model.add(Dense(self.actionSize, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))

        return model

    # Store our outputs in the memory
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    # Decide whether to take action randomly our based on best possible action
    # This step is taken based on epsilon value (exploration rate)
    # "The agent will take the action randomly with %(The value of epsilon)"
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self, batchSize):

        minibatch = random.sample(self.memory, batchSize)

        for state, action, reward, nextState, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(nextState)[0]))
            targetF = self.model.predict(state)
            targetF[0][action] = target

            self.model.fit(state, targetF, epochs=1, verbose=0)

        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


#Create our neural network (Agent)
agent = Agent(stateSize, actionSize)

#Whethet it fails or reach the best outcome we will change done to true
done = False

scores = []

for episode in range(numOfEpisodes):
#Reset the environment
    state = env.reset()
    state = np.reshape(state, [1, stateSize])

    for step in range(numOfSteps):
        # render the einvornment every 100 episode to see our progress
        if episode % 100 == 0:
            env.render()

        # take an action whether randomly or based on previous experiences
        action = agent.act(state)

        nextState, reward, done, info = env.step(action)

        # if the pole falls return a reward with value -10 (punishment)
        reward = reward if not done else -10

        nextState = np.reshape(nextState, [1, stateSize])

        #Store the values for the current action and state
        agent.remember(state, action, reward, nextState, done)

        state = nextState


        #After the pole falls or reach the max number of step print the results
        if done:
            scores.append(step)
            print("episode: {}/{}, score: {}, epsilon: {:.2}".format(episode, numOfEpisodes, step, agent.epsilon))
            break

    if len(agent.memory) > batchSize:
        agent.replay(batchSize)


    #Store the value for the agent each 50 episodes
    if episode % 50 == 0:
        agent.save(outputDir + "weights_" + '{:04d}'.format(episode) + ".hdf5")


