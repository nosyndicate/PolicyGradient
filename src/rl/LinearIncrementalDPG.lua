require 'torch'
require 'nn'

-- this class implement the Linear Deterministic Policy Gradient Method in Continuous Domain
-- Due to the constraint from "linear" and "continuous domain"
-- This is easier for us to put all the thing in this class


local LinearIncrementalDPG, parent = torch.class('rl.LinearIncrementalDPG','rl.Incremental')



function LinearIncrementalDPG:__init(model, optimizer, criticOption, actNum, featureSize, gamma)
	parent.__init(self, model, actNum)
	self.optimizer = optimizer
	self.gamma = gamma or 1
	
	-- for now we are only allow Q learning or Gradient Q learning for critic
	-- other options like batch method may be add in later
	if criticOption == "Q" then
		-- we are using a Linear Critic with Compatible Feature
		-- advantageCritic should have the input same to the row of the Jacobian matrix in compatible feature
		-- however, that row number includes one bias number additional to the feature of the state,
		-- for detail see comments for feature function
		self.modelParametersNum = self.optimizer.params:size()[1]
		self.advantageCritic = nn.Sequential():add(nn.Linear(self.modelParametersNum, 1))
		self.valueFunctionCritic = nn.Sequential():add(nn.Linear(featureSize, 1))
		self.advantageParams, self.advantageGrads = self.advantageCritic:getParameters()
		self.valueFunctionParams, self.valueFunctionGrads = self.valueFunctionCritic:getParameters()
	elseif criticOption ~= "GQ" then
		-- TODO
	else
		assert(false, "unidentified critic option")
	end 
	
	self.criticOption = criticOption
end

-- TODO, add another learning rate for Gradient Q-learning
function LinearIncrementalDPG:setAdditionalLearningRate(alr, vlr)
	self.alr = alr
	self.vlr = vlr
end

function LinearIncrementalDPG:initiateParameters(lower, upper)
	self.optimizer.params:uniform(lower, upper)
end

function LinearIncrementalDPG:getAction(s)
	-- get the desire action for policy
	local desireAction = self.model:forward(s)
		
	
	-- add some noise for exploration 
	local meanTable = desireAction:totable()
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
function LinearIncrementalDPG:setActionStdev(stdev)
	self.stdev = stdev
end

function LinearIncrementalDPG:learn(s, r, sprime)
	if self.criticOption == "Q" then
		self:QCritic(s, r, sprime)
	elseif self.criticOption == "GQ" then
		-- TODO : Need to add in the gradient q learning algorithm
	end
end

function LinearIncrementalDPG:QCritic(s, r, sprime)
	-- since our gradient of the policy may get changed due to the computation of compatible feature,
	-- we have to save that first

	-- first clear out the accumulated gradients	
	self.optimizer.grads:zero()

	local compatibleFeature, jacobian = self:feature(s, self.action)
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
		featureSprime, _ = self:feature(sprime, self.model:forward(sprime))
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

	
	-- there is no easy way to calculate the action gradient of the Approximate Q value Q(s,a)
	-- thus, we are going to do it in the matrix form using the Jacobian Matrix from the compatible feature computation
	-- since we have an bias term in critic parameters w, we have to augment the jacobian matrix by adding another column with 0s
	local temp = torch.Tensor(self.modelParametersNum + 1, self.actNum):fill(0)
	temp:narrow(1, 1, self.modelParametersNum):copy(jacobian)
	local jacobianTranspose = temp:t()
	local actionGradient = torch.mv(jacobianTranspose, self.advantageParams)
	
	-- now actionGradient is a m*1 matrix, m is the dimension of action
	local thetaGradient = torch.mv(jacobian, actionGradient) -- this create a self.modelParametersNum * 1 vector, with gradient of \theta 
	self.optimizer:gradientAscent(thetaGradient)
	
	-- update advantage critic
	advantageGradient:mul(tdError)
	self.advantageParams:add(self.alr, advantageGradient)
	
	-- update value function critic
	valueFunctionGradient:mul(tdError)
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

function LinearIncrementalDPG:feature(s, a)
	local jacobian = torch.Tensor(self.modelParametersNum, self.actNum)
	
	-- at here, we fill the Jacobian matrix,
	-- we do this in a "stupid" way, by doing multiply forward and backward computation for each action dimension
	-- however this way is more general, and can apply to other non linear approximation scheme
	-- nn package has a Jacobian module doing the same task
	for i = 1, self.actNum do
		self.model:forward(s)
		
		-- NOTE : zero the gradient, VERY IMPORTANT
		self.optimizer.grads:zero()
		
		-- use variable selection to determine which gradient value is relevant to this action i
		local selection = torch.Tensor(self.actNum):fill(0)
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
