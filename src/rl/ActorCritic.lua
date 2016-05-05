require 'torch'

-- Implementation of incremental actor critic, using Sarsa critic
-- based on policy gradient lecture of David Silver 
-- http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf


local ActorCritic, parent = torch.class('rl.ActorCritic','rl.PolicySearch')


function ActorCritic:__init(model, actor, optimizer, qCritic, vCritic)
	-- parent method have to be called this way: with dot and pass self as first parameters
	parent.__init(self, model)
	self.actor = actor
	self.qCritic = qCritic
	self.vCritic = vCritic	
	self.optimizer = optimizer
	
	-- learning rate for q critic
	self.beta = 0.001
	
	-- inspect which policy we are using
	
end

function ActorCritic:setLearningRate(alpha)
	parent.setLearningRate(self, alpha)
	self.optimizer:setLearningRate(self.alpha)
end

function ActorCritic:setQCriticLearningRate(beta)
	self.beta = beta
end

function ActorCritic:step(s, r)
	assert(false, "Actor Critic is an incremental method, should not call this method, learn function should be called instead")
end

-- do some work before each episode start
function ActorCritic:startTrial()
	self.action = nil
end

-- do some work after the episode ends
function ActorCritic:endTrial()
	
end

function ActorCritic:getAction(s)
	if not self.action then
		self.action = self:getNextAction(s)
	end
	return self.action
end

function ActorCritic:getNextAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local parameters = self.model:forward(s)
	--print(parameters)
	-- sample from the distribution 
	local action = self.actor:getAction(parameters)
	return action
end

function ActorCritic:learn(s, r, sprime)
	-- this is on policy actor critic, thus, we are using a SARSA critic
	local aprime = self:getNextAction(sprime)
	
	-- compute the quantities for the critic
	local featurePrime = self.actor:compatibleFeature(sprime, aprime)
	local feature = self.actor:compatibleFeature(s, self.action)
	local qSprimeAprime = self.qCritic:foward(featurePrime)
	local qsa = self.qCritic:forward(feature)
	
	-- compute the td error
	local td = r + self.gamma*qSprimeAprime-qsa
	
	-- compute the gradient of the actor parameters
	self.optimizer.grads:zero()
	local dLogPolicyDOutput = self.actor:backward()
	self.model:backward(s, dLogPolicyDOutput)
	-- instead using the direct reward, we use the Q-value
	local tempGradient = torch.cmul(self.optimizer.grads:clone(), qsa)
	
	self.optimizer:gradientAscent(tempGradient)
	
	-- set the next action
	self.action = aprime
end








