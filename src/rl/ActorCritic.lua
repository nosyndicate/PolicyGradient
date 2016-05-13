require 'torch'

-- Implementation of incremental actor critic, using Sarsa critic
-- based on policy gradient lecture of David Silver 
-- http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf


local ActorCritic, parent = torch.class('rl.ActorCritic','rl.Incremental')

-- TODO : fix the stdev blow up problem

--[[
NOTE : 
	For this code work, we are assuming the model is a defined using nn package from torch
	Especially, if we are both adapting the means and stdev for the gaussian policy, we want the 
	model to be build using the ConcatTable 
	
	For example:
	modelMean = nn.Sequential():add(nn.Linear(1, 1))
	modelStdev = nn.Sequential():add(nn.Linear(1, 1)):add(nn.Exp())
	model = nn.ConcatTable():add(modelMean):add(modelStdev)
	
	In this case, the mean and stdev does not share weights in the neural network. Thus, unlike the 
	Deterministic Policy Gradient, we are not gonna using anything other than a linear model
	(I would hope the some other architect would work, but I am not sure how do deal their shared weights)

--]]

function ActorCritic:__init(model, actor, optimizer, featureSize, gamma, adaptiveLearningRate)
	-- parent method have to be called this way: with dot and pass self as first parameters
	parent.__init(self, model, actor.actNum)
	self.actor = actor	
	self.optimizer = optimizer
	self.gamma = gamma or 1
	self.adaptiveLearningRate = adaptiveLearningRate or false	
	self.action = nil
	
	-- if we use adaptiveLearningRate, for the case where actions have different stdev,
	-- we have to mark out which gradient value corresponding to what
	if self.adaptiveLearningRate then
		self.map = {}
		-- we do that by faking a state to do forward and backward computation
		local fakeState = torch.Tensor(featureSize):fill(1)
		
		for i = 1,self.actNum do
			self.model:forward(fakeState)
			
			self.optimizer.grads:zero()
			local selection = torch.Tensor(self.actNum):fill(0)
			selection[i] = 1
			self.model:backward(fakeState, {selection, selection})
			
			local indice = {}
			for j = 1,self.optimizer.grads:nElement() do
				if self.optimizer.grads[j] ~= 0 then
					table.insert(indice, j)
				end
			end
			
			table.insert(self.map, indice)
		end
	end
	
	
	
	-- for now we are SARSA for the critic
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
		
		if not actor.stdev then
			-- if actor is not using a fixed stdev then we have to figure the number of parameter correspond to mean and stdev
			-- NOTE: we always assume the mean module are add in first
			local weights, _ = self.model:get(1):parameters()
			self.meanParamNum = 0
			for i = 1,#weights do
				self.meanParamNum = self.meanParamNum + weights[i]:nElement()
			end
			
			weights, _ = self.model:get(2):parameters()
			self.stdevParamNum = 0
			for i = 1,#weights do
				self.stdevParamNum = self.stdevParamNum + weights[i]:nElement()
			end
		end
		
	elseif torch.type(self.actor) == "rl.SoftmaxPolicy" then
		-- NOTE : have figure out how to do this, yucky!
	else
		assert(false, "unsupported policy")
	end
	
end


function ActorCritic:initiateParameters(lowerBound, upperBound)
	self.optimizer.params:uniform(lowerBound, upperBound)
end


function ActorCritic:setAdditionalLearningRate(alr, vlr)
	self.alr = alr
	self.vlr = vlr
end

function ActorCritic:getAction(s)
	if not self.action then
		self.action = self:getNextAction(s)
	end
	return self.action:totable()
end

-- return the Tensor form for policy update
function ActorCritic:getNextAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local parameters = self.model:forward(s)
	
	-- sample from the distribution 
	local actionTable = self.actor:getAction(parameters)
	
	-- actionTensor is in Tensor form
	local actionTensor = torch.Tensor(actionTable)
	return actionTensor
end




--[[
	This learning algorithm use the linear SARSA
--]]
function ActorCritic:learn(s, r, sprime)
	if torch.type(self.actor) == "rl.GaussianPolicy" then
		self:learnGaussian(s, r, sprime)
	elseif torch.type(self.actor) == "rl.SoftmaxPolicy" then
		-- NOTE : have figure out how to do this, yucky!
	end
end


function ActorCritic:learnGaussian(s, r, sprime)

	-- since our gradient of the policy may get changed due to the computation of compatible feature,
	-- we have to save that first
	-- first clear out the accumulated gradients	
	self.optimizer.grads:zero()
	local dLogPolicyDOutput = self.actor:backward()
	self.model:backward(s, dLogPolicyDOutput)

	local policyGradient = self.optimizer.grads:clone()

	-- since we save the policy gradient with respect to the previous state s and action a
	-- we can start to deal with next state sprime
	-- this is on policy actor critic, thus, we are using a SARSA critic
	local aprime = nil
	if sprime then
		aprime = self:getNextAction(sprime)
	end
	
	
	-- NOTE: when we compute the compatible feature, the forward and backward information
	-- of the module can get overwritten, but the gradient of the actor should be same,
	-- since when only called forward and backward function on model but not on actor
	local compatibleFeature, jacobian = self:gaussianPolicyCompatibleFeature(s, self.action)
	local asa = self.advantageCritic:forward(compatibleFeature)
	local vs = self.valueFunctionCritic:forward(s)
	
	self.advantageGrads:zero()
	self.valueFunctionGrads:zero()
	
	self.advantageCritic:backward(compatibleFeature, torch.Tensor{1})
	self.valueFunctionCritic:backward(s, torch.Tensor{1})
	
	-- these two variables are save for future use
	local advantageGradient = self.advantageGrads:clone()
	local valueFunctionGradient = self.valueFunctionGrads:clone()
	
	local featureSprime, _ = nil, nil
	local asprimea = nil
	local vsprime = nil
	if sprime then
		featureSprime, _ = self:gaussianPolicyCompatibleFeature(sprime, aprime)
		asprimea = self.advantageCritic:forward(featureSprime)
		vsprime = self.valueFunctionCritic:forward(sprime)
	end
	

	local qsa = asa + vs
	
	local tdError = nil
	if sprime then
		local qsprimea = asprimea + vsprime
		tdError = r + self.gamma * qsprimea[1] - qsa[1]
	else
		tdError = r - qsa[1]
	end


	-- Update the policy
	policyGradient:mul(qsa[1])
	
	
	-- NOTE : we multiply another term to fix the gradient norm blow problem, we do this only is stdev is less than 0.5
	-- see Section 4 in paper : Model-Free Reinforcement Learning with Continuous Action in Practice for detail
	if self.adaptiveLearningRate then
		local term = policyGradient:clone():fill(1)
		if self.actor.stdev and self.actor.stdev < 0.5 then
			term:fill(self.actor.stdev):pow(2)
		else
			local parameters = self.model:forward(s)[2] -- get the tensor for stdev of all actions
			for i = 1, self.actNum do
				local indices = self.map[i] -- get the index in the gradient which are related to this action
				for j = 1, #indices do
					if parameters[i] < 0.5 then
						term[indices[j]] = parameters[i]
					end
				end
			end
			term:pow(2)
		end
		policyGradient:cmul(term)
	end
	
	self.optimizer:gradientAscent(policyGradient)
	
	
	-- update advantage critic
	advantageGradient:mul(tdError)
	self.advantageParams:add(self.alr, advantageGradient)
	
	-- update value function critic
	valueFunctionGradient:mul(tdError)
	self.valueFunctionParams:add(self.vlr, valueFunctionGradient)
	
	-- make sure we restore everything
	if sprime then
		self.model:forward(sprime)  -- this restore the state for model
		self.action = aprime -- aprime does not need to restore since we didn't use actor:forward() method
	else
		-- if we don't have sprime, reset it to nil since we reach the end of episode
		self.action = nil
	end
	
	
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
	
	n - number of features
	m - number of action dimensions

	\phi(s) is a n*m matrix, the specific format see the comment for feature function in LinearIncrementalDPG.lua

	\Sigma^{-1} by definition is m*m matrix
	a and \mu(s) by definition is m*1 vector

	thus, \phi(s,a) is n*1 vector
	As we know, the Q(s,a)  = w^{T} * \phi(s,a) is a scalar, thus, w^{T} is a 1*n vector
	
--]]


function ActorCritic:gaussianPolicyCompatibleFeature(state, action)
	local jacobian = torch.Tensor(self.modelParametersNum, self.actNum)
	
	-- at here, we fill the Jacobian matrix,
	-- we do this in a "stupid" way, by doing multiply forward and backward computation for each action dimension
	-- however this way is more general, and can apply to other non linear approximation scheme
	-- note here, unlike the getAction method, we do not use the "actor" in here
	-- nn package has a Jacobian module doing the same task
	local tempOutput = nil	
	
	for i = 1, self.actNum do
		tempOutput = self.model:forward(state)
		
		-- NOTE : zero the gradient, VERY IMPORTANT
		self.optimizer.grads:zero()
		
		-- use variable selection to determine which gradient value is relevant to this action i
		local selection = torch.Tensor(self.actNum):fill(0)
		selection[i] = 1
		
		if self.actor.stdev then
			self.model:backward(state, selection)
		else
			-- if we are adapting stdev, then we need to do this
			self.model:backward(state, {selection, selection})
		end
		
		-- copy the gradient value to the Jacobian column by column
		jacobian:select(2, i):copy(self.optimizer.grads)
	end
	
	-- after we have the Jacobian, we do the following calculation
	-- so \phi(s,a) = \phi(s) * \Sigma^{-1} * (a - \mu(s))
	
	if self.actor.stdev then
		-- actor has a fixed stdev
		-- in this case, the temp is the Tensor of means of all the actions, which is like the deterministic policy
		-- which is easier
		local diff = torch.csub(action, tempOutput)
		-- this equal we times a diagnal matrix with actor.stdev 
		diff:div(self.actor.stdev)
		local feature = torch.mv(jacobian:clone(), diff)

		return feature, jacobian
	else
		-- now we deal with the case where we have to mean and stdev
		-- unlike the case above, we do this action by action, we compute the scalar value for mean and variance, when we multiply it
		-- matrix element-wise
		
		local feature = jacobian:clone()
		local temp = torch.Tensor(self.modelParametersNum):fill(0)
		for i = 1, self.actNum do
			local mu = tempOutput[1][i]
			local stdev = tempOutput[2][i]
			
			-- for these correction term, look gaussian policy
			local meanCorrection = (action[1]-mu)/(stdev*stdev)
			local stdevCorrection = (action[1]-mu)*(action[1]-mu)/(stdev*stdev) - 1
			
			local feature = jacobian:clone()
			
			feature:select(2, i):narrow(1, 1, self.meanParamNum):mul(meanCorrection)
			feature:select(2, i):narrow(1, 1 + self.meanParamNum, self.stdevParamNum):mul(stdevCorrection)
			
			-- combine all the column into one, since we are using linear model, this will match perfectly and no interferer between each other 
			temp:add(feature:select(2,i):clone())
			
		end
		
		feature = temp
		return feature, jacobian
	end
end





