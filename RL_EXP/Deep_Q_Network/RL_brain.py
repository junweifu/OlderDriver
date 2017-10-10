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
import tensorflow as tf

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

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
            ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2)))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        
    def _build_net(self):
        #------build evaluate_net-------
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='Q_target')
        with tf.variable_scope('eval_net'):
        c_names,n_l1,w_initializer,b_initializer = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],10,tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
            b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
            l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2',[n_l1,n_actions],initializer=w_initializer,collections=c_names)
            b2 = tf.get_variable('b2',[1,n_actions],initializer=b_initializer,collections=c_names)
            self.q_eval = tf.matmul(l1,w2)+b2    
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
            
        #------build target_net---------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hsstack((s,[a,r],s_))
        
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index,:] = transition
        
        self.memory_counter+=1
        
    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.uniform <self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assigno(t,e) for t,e in zip(t_params,e_params)])
        
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter ==0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')
            
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size,size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            
            q_next,q_eval = self.sess.run([self.q_next,self.q_eval],
                                          feed_dict={
                                              self.s_:batch_memory.iloc[:,-self.n_features:],
                                              self.s:batch_memory.iloc[:,:self.n_features]
                                          })
            
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size,dtype=np.int32)
            eval_act_index = batch_memory[:,self.n_features].astype(int)
            reward = batch_memory[:,self.n_features+1]
            
            q_target[batch_index,eval_act_index] = reward + self.gamma*np.max(q_next,axis=1)
            
            _,self.cost = self.sess.run([self._train_op,self.sess],
                                        feed_dict={self.s:batch_memory[:,:self.n_features],self.q_target:q_target}
                                        )
            self.cost_his.append(self.cost)
            
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon <self.epsilon_max else self.epsilon_max
            self.learn_step_counter+=1
        
        def plot_cost(self):
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(self.cost_his)), self.cost_his)
            plt.ylabel('Cost')
            plt.xlabel('training steps')
            plt.show()
    