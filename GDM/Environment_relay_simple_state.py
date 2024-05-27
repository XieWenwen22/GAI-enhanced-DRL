import copy
import math
import random

import numpy as np
import warnings
from gym import Env
from gym.spaces import Discrete, Box, Dict

warnings.filterwarnings("ignore")

class Environment(Env):
    def __init__(self):
        self.N = 50
        self.action_space = Box(low=np.array([-1]*(2+2)), high=np.array([1]*(2+2)))
        self.observation_space = Box(low=np.array([0]*(3+2+self.N)), high=np.array([1]*(3+2+self.N)))
        self.UAV1_location = np.array([1,0,6])
        self.UAV2_location = np.array([1,0,6])
        self.User_location = np.array([[150,200,0],[280,200,0]])
        self.UAV2_boundary_min = 0
        self.UAV2_boundary_max = 300
        self.T = 100
        self.center_location = np.array([0,0,0])
        self.wavelength = (3*10**8)/(80*10**9)
        self.d = self.wavelength/2
        self.near_field = 2*(0.5)**2/self.wavelength
        self.rou = 10**(-72/10)
        self.P = 10**(30/10) #30dbm
        self.P_BS = 1000
        self.P_UAV = 1000
        self.sigma = 10**(-80/10) #-80dbm
        self.alpha = 10**(-50/10)

    def reset(self):
        self.UAV1_location = np.array([1, 0, 6])
        # self.UAV2_location = np.array([1, 0, 6])
        H = self.channel_near_filed(self.UAV1_location)
        return np.concatenate((self.UAV1_location,H,np.array([0]),np.array([0])))


    def step(self, action,i_episode,t,flag_1):

        UAV1_distance = action[:2] * 5
        BS_power = (action[2] - (-1))/2 * self.P_BS
        UAV1_power = ((action[3] - (-1))/2) * self.P_UAV
        self.UAV1_location[:2] = self.UAV1_location[:2] + UAV1_distance

        rate_source, channel = self.rate_near_field(self.UAV1_location,BS_power)
        rate_UAV1 = self.rate_far_field_Users(self.UAV1_location,self.User_location[0],UAV1_power)

        rate = rate_UAV1
        reward = rate*10 - (BS_power+UAV1_power)*10**(-3)*5
        reward_original = copy.deepcopy(reward)

        # constraints

        # location
        if (not (self.UAV2_boundary_min < self.UAV1_location[0] < self.UAV2_boundary_max)) or (not (self.UAV2_boundary_min < self.UAV1_location[1] < self.UAV2_boundary_max)):
            reward -=50
        if math.dist(self.UAV1_location,self.center_location) > self.near_field:
            reward -= 50
        if rate_source <= rate_UAV1:
            reward -= 50


        state = np.concatenate((self.UAV1_location,channel,np.array([rate_source]),np.array([rate_UAV1])))
        info = {}
        done = 1
        return state, reward, info, done, rate, (BS_power+UAV1_power)


    def channel_near_filed(self,UAV1_location):
        antenna_distance = np.zeros(self.N)
        b = np.zeros(self.N)
        r = math.dist(UAV1_location,self.center_location)
        for i in range(1,self.N+1):
            antenna_distance[i-1] = math.dist(UAV1_location,np.array([0,((2*i-self.N-1)/2)*self.d,0]))
            b[i-1] = np.exp(-1j*(2*np.pi/self.wavelength)*(antenna_distance[i-1]-r))
        b = (1/np.sqrt(self.N))*b.T
        H = np.sqrt(self.N)*((np.sqrt(self.rou)/r)*np.exp(-1j*2*(np.pi/self.wavelength)*r))*b
        return H

    def rate_near_field(self,UAV1_location,BS_power):
        H = self.channel_near_filed(UAV1_location)
        action = H/abs(H)
        return np.log2(1+BS_power*abs(action @ H)**2/self.sigma), H

    def rate_far_field_Users(self,source_loaction,destination_location,source_power):
        H1 = self.alpha/((math.dist(source_loaction,destination_location))**2+0.000001)

        return np.log2(1+source_power*H1/self.sigma)