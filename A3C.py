###TODO

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


environment=Simulator()

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        space = environment.action_space()[0] * environment.action_space()[1]
        self.layer1 = nn.Linear(space * 2, space)
    
    def forward (self, input):
        None