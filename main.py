"""
imports:
"""
# import numpy as np
# import gym
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
#
# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

from selenium import webdriver
from constants import *

from selenium_helper import ScreenCapture
from selenium.webdriver.chrome.options import Options

ENVIRONMENT_NAME = 'Achtung-v1.0'

s = ScreenCapture()
options = Options()
options.add_experimental_option("prefs",CHROME_PREFERENCES)
driver = webdriver.Chrome(executable_path='C:/bin/chromedriver.exe', options=options)
s.set_flash(driver)
