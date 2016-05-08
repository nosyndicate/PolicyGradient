require 'torch'
require 'nn'

-- this class implement the Linear Deterministic Policy Gradient Method in Continuous Domain
-- Due to the constraint from "linear" and "continuous domain"
-- This is easier for us to put all the thing in this class


local LinearIncrementalDeterministicPolicy, parent = torch.class('rl.LinearDeterministicPolicy','rl.Incremental')



function LinearIncrementalDeterministicPolicy:__init(model, optimizer, criticOption, actDim, featureSize, gamma)
	self.model = model
	self.optimizer = optimizer
	self.actDim = actDim
	self.gamma = gamma or 1
	
	-- for now we are only allow Q learning or Gradient Q learning for critic
	-- other options like batch method may be add in later
	if criticOption == "Q" then
		-- we are using a Linear Critic with Compatible Feature
		-- advantageCritic should have the input same to the row of the Jacobian matrix in compatible feature
		-- however, that row numer includes one bias number additional to the feature of the state,
		-- for detail see comments for feature function
		self.modelParametersNum = self.optimizer.params:size()[1]
		self.advantageCritic = nn.Sequential(nn.Linear(self.modelParametersNum, 1))
		self.valueFunctionCritic = nn.Sequential(nn.Linear(featureSize, 1))
		self.advantageParams, self.advantageGrads = self.advantageCritic:getParameters()
		self.valueFunctionParams, self.valueFunctionGrads = self.valueFunctionCritic:getParameters()
	elseif criticOption ~= "GQ" then
		-- TODO
	else
		assert(false, "unidentified critic option")
	end 
	
	self.criticOption = criticOption
end

-- TODO, add compatiblility for Gradient Q-learning
function LinearIncrementalDeterministicPolicy:setAdditionalLearningRate(alr, vlr)
	self.alr = alr
	self.vlr = vlr
end

function LinearIncrementalDeterministicPolicy:getAction(s)
	-- get the desire action for policy
	self.desireAction = self.model:forward(s)
	-- add some noise for exploration 
	local meanTable = self.desireAction:totable()
	local actionTable = {}
	
	for i=1,#meanTable do
		actionTable[i] = torch.normal(meanTable[i], self.stdev)
	end
	
	-- store the action in tensor for backward computation
	self.action = torch.Tensor(actionTable)
	
	-- return the action table
	return actionTable
end

-- this is used for set the stdev of the behavior policy so the agent have exploration
function LinearIncrementalDeterministicPolicy:setActionStdev(stdev)
	self.stdev = stdev
end

function LinearIncrementalDeterministicPolicy:learn(s, r, sprime)
	if self.criticOption == "Q" then
		self:QCritic(s, r, sprime)
	elseif self.criticOption == "GQ" then
		-- TODO : Need to add in the gradient q learning algorithm
	end
end

function LinearIncrementalDeterministicPolicy:QCritic(s, r, sprime)
	-- since we are gradient of the policy may get changed due to the computation of compatible feature,
	-- we have to save that first

	-- first clear out the accumulated gradients	
	self.optimizer.grads.zero()
	-- since we don't have gradsOutput for the second parameter, we use (1 1 1 1 ... 1) instead
	self.model:backward(s, torch.Tensor(self.actDim):fill(1))
	local gradient = self.optimizer.grads:clone()

	local compatibleFeature, jacobian = self:feature(s, self.action)
	local asa = self.advantageCritic:forward(compatibleFeature)
	
	local featureSprime, _ = self:feature(sprime, self.model:forward(sprime))
	local asprimea = self.advantageCritic:forward(featureSprime)
	
	local vs = self.valueFunctionCritic:forward(s)
	local vsprime = self.valueFunctionCritic:forward(sprime)
	
	local qsa = asa + vs
	local qsprimea = asprimea + vsprime
	
	local tdError = r + self.gamma * qsprimea - qsa
	
	local
	self.optimizer:gradientAscent(gradient)
	
	local advantageGradient = compatibleFeature:mul(tdError)
	self.advantageParams:add(self.alr, advantageGradient)
	
	local valueFunctionGradient = s:clone():mul(tdError)
	self.valueFunctionParams:add(self.vlr, valueFunctionGradient)

end

--[[
 	This function return the compatible feature \phi(s,a)
	The computation of the compatilbe feature is a little tricky in multidimensional action cases
	Suppose we have *m* action dimension, then our action a is a m*1 vector, and \mu_{\theta}(s) has the same dimension as action, which is a m*1 vector
	According to the definition, (section 4.3) in paper Deterministic Policy Gradient Algorithm
	              d \mu_{\theta}(s)
	\phi(s,a) = ---------------------- * (a - \mu_{\theta}(s))
	                  d \theta
	                  
	in this equation, the first term in right hand side is a partial derivative with respect to \theta
	this is a n*m Jacobian matrix (according to the text in paper), and the last term on RHS is a m*1 vector as discussed above
	which suggest \mu_{\theta}(s) is a vector valued function where \theta is a n*1 vector, \phi(s) is n*m matrix,
	and \mu_{\theta}(s) = \phi(s)^{T} * \theta
	
	If, for example, we have two state feature variable s_1 and s_2 for state s and our action dimension is three
	Then, the matrix \phi(s) is like this
	
		s_1  0   0
		s_2  0   0
    	 0	s_1  0
    	 0  s_2  0
    	 0   0  s_1
    	 0   0  s_2 
   	
   	and \theta vector is the column vector (\theta_1, \theta_2, \theta_3, \theta_4, \theta_5 \theta_6)^{T}
   	in this case, we can use the a Linear layer in nn package nn.Linear(2, 3) to compute \mu_{\theta}(s)
   	
   	The matrix showed above is also the Jacobian matrix in the equation above, and this matrix is also apply to the stochastic policy gradient
--]]

function LinearIncrementalDeterministicPolicy:feature(s, a)
	local jacobian = torch.Tensor(self.modelParametersNum, self.actDim)
	
	-- at here, we fill the Jacobian matrix,
	-- we do this in a "stupid" way, by doing multiply forward and backward computation for each action dimension
	-- however this way is more general, and can apply to other non linear approximation scheme
	for i = 1, self.actDim do
		self.model:forward(s)
		
		-- NOTE : zero the gradient, VERY IMPORTANT
		self.optimizer.grads.zero()
		
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
	
	return feature
end
