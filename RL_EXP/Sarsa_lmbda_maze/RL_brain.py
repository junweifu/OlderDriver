# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:51:21 2017

@author: Administrator
"""

"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space#a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)
        
    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
                
    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.rand()<self.epsilon:
            state_action = self.q_table.ix[observation,:]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action
    
    def learn(self,*args):
        pass

#off policy
class QLearningTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(QLearningTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

#on policy
class SarsaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s,a]
        if s_ != 'terminal':
            q_target = r+self.gamma*self.q_table.ix[s_,a_]
        else:
            q_target = r
        self.q_table.ix[s,a] += self.lr * (q_target - q_predict)
        
class SarsaLambdaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
        super(SarsaLambdaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            # append new state to q table
            self.q_table = self.q_table.append(to_be_append)
            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.ix[s_,a_]
        else:
            q_target = r
        error = q_target - q_predict
        
        # self.eligibility_trace.ix[s, a] += 1
        
        self.eligibility_trace.ix[s,:] *= 0
        self.eligibility_trace.ix[s,a] = 1
        
        self.q_table += self.lr*error*self.eligibility_trace
        
        self.eligibility_trace *= self.gamma*self.lambda_
        print("lambda_",self.lambda_)
        print("gamma",self.gamma)
        print("eligibility_trace",self.eligibility_trace)











    