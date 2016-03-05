require 'torch'

-- create a global table for learner package
rl = {};

torch.include('rl','Learner.lua');
torch.include('rl','PolicySearch.lua');
torch.include('rl','Reinforce.lua');
torch.include('rl','GPOMDP.lua');
torch.include('rl','Policy.lua');
torch.include('rl','SoftmaxPolicy.lua');
torch.include('rl','Optimizer.lua');
torch.include('rl','StochasticGradientDescent.lua');







