

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import oscillator_cpp
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box

import tensorflow as tf

import pandas as pd




import csv



       











class oscillatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, len_state=250, ep_length=1000, nosc=1000, epsilon=0.03, frrms=0.1,random=False, #ep_length=10000
                sigmoid=False):

        """ 
        Init function:
        sigmoid: Function that we observe instead of original one: Bool
        len_state: shape of state that agent observes [250,1]: integer
        BVDP params
        nosc: number of oscillators: integer
        epsilon: coupling parameter: float
        frrms: width of the distribution of natural frequencies: float
        """

        super(oscillatorEnv, self).__init__()
        #Call init function and save params
        if random:
            epsilon = np.random.uniform(0.03,0.5)
        self.y = oscillator_cpp.init(nosc,epsilon,frrms)
        self.nosc = nosc
        self.epsilon = epsilon
        self.frrms = frrms
        
        self.ep_length = ep_length

        #Dimensionality of our observation space
        self.dim = 1
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1.5, high=1.5, shape=(len_state,), dtype=np.float32)
        self.observation_size = 250

        #Meanfield for all neurons
        self.x_val = oscillator_cpp.Calc_mfx(self.y)
        self.y_val = oscillator_cpp.Calc_mfy(self.y)

        #Episode Done?
        self.done = False
        self.current_step = 0 

        #Our current state, with length(1,len_state)
        self.y_state = []
        self.x_state = []
        #Our actions 
      
        
        self.len_state = len_state
        
       

        self.total_reward = []
        self.actions = []

        self.all_states = []

        self.num_episode = 0
        #Reset environment
        self.reset()

        
     







    def step(self, action):
        """
        Function that called at each step.
        action: signal to make perturbation: [[float]]
        returns: arrayed_version:np.array(1,len_state), 
        Reward: Our reward function :float, 
        done: Does it end? :Bool, 
        additional_information: Nothing to show :( :{} 
        """
      
        #Vectorized form for stable baselines
        
        val = float(action[0])
        self.actions.append(val)

        self.y = oscillator_cpp.Pertrubation(self.y, val)
        self.y = oscillator_cpp.Make_step(self.y)
        
            
        #Calculate MeanField
        self.x_val = oscillator_cpp.Calc_mfx(self.y)
        self.y_val = oscillator_cpp.Calc_mfy(self.y)
        
        #if sigmoid:
            #self.x_val = sigmoid(self.x_val)
            #self.y_val = sigmoid(self.y_val)
        

        # noise = np.random.normal(0, 0.4, 1)
        #Save our state
        self.y_state.append(self.y_val)
        self.x_state.append(self.x_val)#        self.x_state.append(self.x_val+noise)

        
        #Check length of our state
        if len(self.y_state) > self.len_state:
            self.y_state = self.y_state[1:]
            self.x_state = self.x_state[1:]

        self.current_step += 1

        self.done = self.current_step >= self.ep_length

        #Make vectorized form
        arrayed_version = np.array(self.y_state)
        
        #if sigmoid:
            #arrayed_version = sigmoid(arrayed_version)

        # print('arrayed_version: ', arrayed_version)
        # print(arrayed_version.shape)
        # print('x_val: ', self.x_val)
        # print('x_state: ', self.x_state )
        # print(len(self.x_state)

        # self.done = False

        rew = self.Reward(self.x_val,self.x_state,val)#rew = self.Reward(self.x_val+noise,self.x_state,val)

     

        
        self.total_reward.append(rew)

        self.all_states.append([arrayed_version])

       

            
        return arrayed_version, rew, self.done, {} 




  
    def reset(self):
        """
        Reset environment, and get a window 250 of self.len_state size
        Returns:arrayed_version:np.array(1,len_state)

        """

 
        

      
      


        self.current_step = 0 
        self.y_state = []
        self.x_state = []
        self.y = oscillator_cpp.init(self.nosc,self.epsilon,self.frrms)

        for i in range(self.len_state):
            oscillator_cpp.Make_step(self.y)
            
            self.x_val = oscillator_cpp.Calc_mfx(self.y)
            self.y_val = oscillator_cpp.Calc_mfy(self.y)

            self.y_state.append(self.y_val)
            self.x_state.append(self.x_val)
            
            #Check length of our state
            if len(self.y_state) > self.len_state:
                self.y_state = self.y_state[1:]
                self.x_state = self.x_state[1:]

        arrayed_version = np.array(self.y_state)    
        
        #if sigmoid:
            #arrayed_version = sigmoid(arrayed_version)
        
        self.num_episode += 1
        
        return arrayed_version
        
    def render(self, mode='human', close=False):
        """
        Pass...
        """
        pass

    def Reward(self, x,x_state,action_value,baseline = False):
        """
        Super duper reward function, i am joking, just sum of absolute values which we supress + penalty for actions
        returns: float
        """
        
        return -(x-np.mean(x_state))**2 - 2*np.abs(action_value)

        

    def sigmoid(x):
        x_0 = 1.2
        return 1. / (1. + np.exp(-x/x_0))



    #@staticmethod
    def cost_fn(self, actions, states):
        # is_tf = tf.compat.v1.contrib.framework.is_tensor(states)

     
        print('actions shape: ', actions.shape)
        scores = tf.zeros(actions.get_shape()[0].value) #if is_tf else np.zeros(actions.shape[0])


        # reward = tf.exp(-1*tf.losses.mean_squared_error(next_states, setpoints_)*1)




        # mae = tf.keras.losses.MeanAbsoluteError()


        # reward = tf.exp(-1*mae(next_states, setpoints))


        # reward = tf.exp(-1*tf.math.reduce_sum(tf.math.pow((next_states-setpoints),2)*20))
        # print('state: , state')
        # print('loss: ', loss)

        print('states: ', states.shape)

        single_state = states[:, -1:]
        mean_states = tf.reshape(tf.math.reduce_mean(states, axis=1), [-1, 1])

        print('single state: ', single_state.shape)
        print('mean state: ', mean_states.shape)


        # scores = -(x-np.mean())**2 - 2*np.abs(actions)

        # state_com = tf.reshape(tf.losses.mean_squared_error(single_state, mean_states), [-1,1])


        state_com = tf.losses.mean_squared_error(single_state, mean_states)



        action_com = tf.reshape(0*tf.math.abs(actions), [-1,])




        print('state com: ', state_com.shape)
        print('action : ', action_com.shape)

        scores = -tf.exp(-state_com - action_com)


        # np.exp(-(x-np.mean(x_state))**2 - 2*np.abs(action_value))


        print('score: ', scores, scores.shape)


      

        return scores

    
