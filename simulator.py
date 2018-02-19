import numpy as np
import os
import random
import math
from sklearn.metrics import accuracy_score
import scipy.io.wavfile
from classifiers.basicMFCC.feature import audio2feature
from classifiers.basicMFCC.classifier import CustomClassifier

from concurrent.futures import ProcessPoolExecutor

class ParalellSyncronSimulator:
    #TODO:test
    def __init__(self, simulator_num):
        self.pool = ProcessPoolExecutor(max_workers=simulator_num)
        self.simulators = []
        for i in range(simulator_num):
            self.simulators.append(Simulator())
    def step(self,actions):
        futures = []
        for i in range(len(actions)):
            futures.append(self.pool.submit(self.simulators[i].par_step,actions[i]))
        result =[[],[],[]]
        for i in range(len(actions)):
            sim, (next_state, reward, done) =futures[i].result()
            self.simulators[i] = sim
            result[0].append(next_state)
            result[1].append(reward)
            result[2].append(done)
        return np.array(result[0]), np.array(result[1]), np.array(result[2])    
    
    def reset(self):
        result =[]
        for i in range(len(self.simulators)):
            result.append(self.simulators[i].reset())
        return result




class Simulator:

    def __init__(self, real_simulation = False):
        #if it is true= state is determenid using only used datapoints
        self.real_simulation = real_simulation #TODO: do the step modifications
        # parameters
        self.enough_accuracy = 0.8  # if it achieves this accuracy, the episode ends
        # when number of samples is more than the #person*this number, then end of episode
        self.max_sample_per_person = 24
        self.accuracy_reward_coef = 60  # how much reward it get when as accuracy improves
        self.word_penalty = 2.0  # penalty for each new word
        self.goal_reward = 20 # reward for reaching the given accuracy
        # END parameters
        self.model = CustomClassifier()
        self.learning_curve=[]
        self.__loadData()
        self.reset()

    def __loadData(self, path='data'):
        self.data = []  # array of users-->array of words-->array of samples
        users = [f for f in os.listdir(path) if not f[0] == '.']
        self.nb_users = len(users)
        for directory in users:
            self.data.append([])
            words = [f for f in os.listdir(
                os.path.join(path, directory)) if not f[0] == '.'][:2]   #TODO: remove.. csak 1 szót engedek
            self.nb_word = len(words)
            for subdirectory in words:
                self.data[-1].append([])
                samples = [f for f in os.listdir(os.path.join(
                    path, directory, subdirectory)) if not f[0] == '.']
                self.nb_samples = len(samples)  # samples per word
                for file in samples:

                    rate,data =scipy.io.wavfile.read(os.path.join(path,directory,subdirectory,file))
                    self.data[-1][-1].append(audio2feature(data,rate))

        #wrapper for paralell execution
    def par_step(self,action):
        return self, self.reset()

    def reset(self):
        self.train_X = []
        self.train_Y = []
        self.used = set()
        self.history = [1.0/self.nb_users]
        self.count_items=np.zeros((self.nb_users,self.nb_word))
        self.learning_curve.append(0)
        #return first observation
        return np.zeros((2,self.nb_users,self.nb_word))

    #wrapper for paralell execution
    def par_step(self,action):
        return self, self.step(action)
       
    # action: tuple-->(user_id, word_id)
    def step(self, action):
        # generate sample and it in the dataset.
        samples = [x for x in range(self.nb_samples - 1)]
        random.shuffle(samples)
        while 0 < len(samples):
            if not((action[0], action[1],samples[0]) in self.used):
                self.used.add((action[0], action[1],samples[0]))
                self.train_X.append(self.data[action[0]][action[1]][samples[0]])
                self.train_Y.append(action[0])
                samples=[]
            else:
                samples.pop(0)


        self.model.fit(self.train_X, self.train_Y)

        observation=[]
        # make observation
        full_temp_X=[]
        full_temp_Y=[]

        for iduser, user in enumerate(self.data):
            observation.append([])
            for idword, word in enumerate(user):
                temp_X=[]
                temp_Y=[]
                for example in range(len(word)):
                    if not((iduser, idword, example) in self.used):
                        temp_X.append(word[example])
                        temp_Y.append(iduser)
                        #for aggregated accuracy
                        full_temp_X.append(word[example])
                        full_temp_Y.append(iduser)
                pred=self.model.predict(temp_X)
                observation[-1].append(accuracy_score(temp_Y, pred))

        #accuracy calculation
        predicted = self.model.predict(full_temp_X)
        self.history.append(accuracy_score(full_temp_Y, predicted))
        
        # REWARD calculation
        reward=-self.word_penalty + self.accuracy_reward_coef * \
           (self.history[-1] - self.history[-2])
        if self.history[-1] >= self.enough_accuracy:
            done=True
            reward += self.goal_reward
        elif len(self.history) > self.max_sample_per_person * self.nb_users:
            done=True
        else:
            done=False

        #account word number: squeeze between 0 and 1 with tanh
        self.count_items[action[0], action[1]] = self.count_items[action[0], action[1]] + 1.0/self.nb_samples

        observation=np.array(observation).reshape((1,self.nb_users,self.nb_word))
        observation= np.concatenate((observation,self.count_items.reshape((1,self.nb_users,self.nb_word))),axis=0)
        #save reward:
        self.learning_curve[-1] = reward
        return observation, reward, done

    def action_space(self):  # tuple of available actions
        return (self.nb_users, self.nb_word)

    def uniform_sample(self):
        return (random.randrange(0, self.nb_users), random.randrange(0, self.nb_word))


    def gethistory(self):
        return self.history

    def area_under_curve(self, length=-1):
        """ Laplace smoothed area under the curve value"""
        sum=0
        for idx, item in enumerate(self.history):
            if (length == -1 or idx < length):
                sum += item
        if length == -1:
            return sum / (len(self.history))
        else:
            return sum / min(length, len(self.history))


    def augment_state(self,observation):
        #TODO
        #generate order numbers
        nr=[x for x in range(self.nb_users)]
        nr=np.array(nr)
        observation=np.append(observation,nr,axis=0)
        np.random.shuffle(observation)
        order = observation[-1,:]
        observation=observation[:-1,:]
        return observation, order

