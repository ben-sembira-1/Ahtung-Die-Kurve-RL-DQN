import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ahtungGame.py import *
import pygame



class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      number_of_players = 4
      #   number_of_players = ord(input("How many players?"))
      ]
      
  def step(self, action):
    pass
  def reset(self):
    pass
  def render(self, mode='human', close=False):
    pass
