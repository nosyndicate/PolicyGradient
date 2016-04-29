require 'torch'

-- create a global table for learner package
rl = {}

-- policy gradient methods
torch.include('rl','Learner.lua')
torch.include('rl','PolicySearch.lua')
torch.include('rl','Reinforce.lua')
torch.include('rl','GPOMDP.lua')


-- policy representation
torch.include('rl','Policy.lua')
torch.include('rl','SoftmaxPolicy.lua')
torch.include('rl','GaussianPolicy.lua')

-- feature representation
torch.include('rl','Feature.lua')
torch.include('rl','TileCoding.lua')

-- optimizer
torch.include('rl','Optimizer.lua')
torch.include('rl','StochasticGradientDescent.lua')


-- value functions
torch.include('rl','Incremental.lua')


-- bounding the parameters
torch.include('rl', 'InvertingGradient.lua')




