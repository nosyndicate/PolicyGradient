require 'torch'
require 'nn'




local DDPG, parent = torch.class('rl.DDPG','rl.Incremental')



function DDPG:__init(model, optimizer, criticOption, actNum, featureSize, gamma)
	parent.__init(self, model, actNum)
	
end


function DDPG:setAdditionalLearningRate(alr, vlr)
	self.alr = alr
	self.vlr = vlr
end

function DDPG:initiateParameters(lower, upper)
	self.optimizer.params:uniform(lower, upper)
end

function DDPG:getAction(s)
	
end

-- this is used for set the stdev of the behavior policy so the agent have exploration
function DDPG:setActionStdev(stdev)
	self.stdev = stdev
end

function DDPG:learn(s, r, sprime)
	
end

