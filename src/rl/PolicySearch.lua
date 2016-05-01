require 'torch'

local PolicySearch, parent = torch.class('rl.PolicySearch','rl.Learner')


function PolicySearch:__init(model, actor, optimizer)
	-- parent method have to be called this way: with dot and pass self as first parameters
	parent.__init(self, model)
	self.actor = actor
	self.optimizer = optimizer
end

function PolicySearch:initiateParameters(lowerBound, upperBound)
	self.optimizer.params:uniform(lowerBound, upperBound)
end

function PolicySearch:setLearningRate(alpha)
	parent.setLearningRate(self, alpha)
	self.optimizer:setLearningRate(self.alpha)
end

function PolicySearch:getAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local parameters = self.model:forward(s)
	-- sample from the distribution 
	local action = self.actor:getAction(parameters)
	return action
end

function PolicySearch:learn(s, r)
	local gradient = self:calculateGradient(s, r)
		
	-- update the parameters with the gradient
	self.optimizer:gradientAscent(gradient)
end

function PolicySearch:calculateGradient(s, r)
	
end






