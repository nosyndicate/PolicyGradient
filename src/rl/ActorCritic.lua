require 'torch'

-- Implementation of incremental actor critic, using Sarsa critic
-- based on policy gradient lecture of David Silver 
-- http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf


local ActorCritic, parent = torch.class('rl.ActorCritic','rl.PolicySearch')


function ActorCritic:__init(model, actor, optimizer)
	-- parent method have to be called this way: with dot and pass self as first parameters
	parent.__init(self, model)
	self.actor = actor	
	self.optimizer = optimizer
	self.gamma = gamma or 1
	
	-- for now we are sarsa for the critic
	-- other options like batch method may be add in later

	-- we are using a Linear Critic with Compatible Feature
	-- advantageCritic should have the input same to the row of the Jacobian matrix in compatible feature
	-- however, that row number includes one bias number additional to the feature of the state,
	-- for detail see comments for compatible feature functions
	if torch.type(self.actor) == "rl.GaussianPolicy" then
		self.modelParametersNum = self.optimizer.params:size()[1]
		self.advantageCritic = nn.Sequential():add(nn.Linear(self.modelParametersNum, 1))
		self.valueFunctionCritic = nn.Sequential():add(nn.Linear(featureSize, 1))
		self.advantageParams, self.advantageGrads = self.advantageCritic:getParameters()
		self.valueFunctionParams, self.valueFunctionGrads = self.valueFunctionCritic:getParameters()
	elseif torch.type(self.actor) == "rl.SoftmaxPolicy" then
		-- TODO : add compatible feature for softmax policy
	else
		assert(false, "unidentified policy")
	end
	
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



--[[
	We derive the compatible feature for the multivariate Gaussian policy using the multivariate Gaussian distribution

	https://en.wikipedia.org/wiki/Multivariate_normal_distribution 
	
	where x is the action and \mu is parameterized by \theta (\mu = \phi(s)^{T} * \theta)

	See two papers for details: 
	1. policy gradient theorem https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf	
	2. deterministic policy gradient http://jmlr.org/proceedings/papers/v32/silver14.pdf 

	Result:
	compatible feature \phi(s,a) = \phi(s) * \Sigma^{-1} * (a - \mu(s))
	This derivation using Equation 84 from the matrix handbook 
	
	n -  number of features
	m - number of action dimensions

	\phi(s) is a n*m matrix, the specific format see the comment for feature function in LinearIncrementalDPG.lua

	\Sigma^{-1} by definition is m*m matrix
	a and \mu(s) by definition is m*1 vector

	thus, \phi(s,a) is n*1 vector
	As we know, the Q(s,a)  = w^{T} * \phi(s,a) is a scalar, thus, w^{T} is a 1*n vector
	
--]]


function ActorCritic:gaussianPolicyCompatibleFeature(state, action)
	local modelParametersNum = nil
	local jacobian = torch.Tensor(self.modelParametersNum, self.actDim)
	
	-- at here, we fill the Jacobian matrix,
	-- we do this in a "stupid" way, by doing multiply forward and backward computation for each action dimension
	-- however this way is more general, and can apply to other non linear approximation scheme
	-- nn package has a Jacobian module doing the same task
	for i = 1, self.actDim do
		self.model:forward(s)
		
		-- NOTE : zero the gradient, VERY IMPORTANT
		self.optimizer.grads:zero()
		
		-- use variable selection to determine which gradient value is relevant to this action i
		local selection = torch.Tensor(self.actDim):fill(0)
		selection[i] = 1
		self.model:backward(s, selection)
		
		-- copy the gradient value to the Jacobian column by column
		jacobian:select(2, i):copy(self.optimizer.grads)
	end
	
	-- after we have the Jacobian, we multiply it with the action vector
	-- so \phi(s,a) = jacobian * a
	-- note that in the comment above, the equation is jacobian * (a-\mu)
	-- however, due to the off-policy situation, we cannot substract \mu from a
	-- this is because the compatible feature for state s' will become 0 since a = \mu in state s'
	
	local feature = torch.mv(jacobian, a)
	
	return feature, jacobian
end





