import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=SequentialMemory(limit=50000, window_length=1),
               nb_steps_warmup=10, target_model_update=1e-2, policy=EpsGreedyQPolicy())

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
