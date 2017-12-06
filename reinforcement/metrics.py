import torch
from itertools import count
import matplotlib.pyplot as plt


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Metrics:
    def __init__(self, environment):
        self.environment = environment


    def compare_policy(self,policylist = [], testnumber = 10, plot = False):
        policylist.append(("random",self.random_policy))
        results=[]
        for i in range(testnumber):
            results.append([])
            for name, algo in policylist:
                state = self.environment.reset()
                state = Tensor(state.reshape(1,2*self.environment.nb_users*self.environment.nb_word))
                for t in count():
                    next_state, reward, done = self.environment.step(algo(state))
                    state = Tensor(next_state.reshape(1,2*self.environment.nb_users*self.environment.nb_word))
                    if done:
                        break
                results[-1].append(self.environment.gethistory())
                if plot:
                    plt.plot(self.environment.gethistory(),label=name)
            if plot:
                plt.legend(loc='best', ncol=2, mode="expand", fancybox=True)
                plt.show()
            
        #calculate metric
        #increase
        increase = []
        for i in range(len(policylist)):
            increase.append(0.0)
            for j in range(testnumber):
                increase[-1] = increase[-1] + (results[j][i][-1] / len(results[j][i]))
            increase[-1] = increase[-1] / testnumber
        #Area Under Curve: TODO
        return increase
        
    
    def random_policy(self,state=None):
        return self.environment.uniform_sample()

    
