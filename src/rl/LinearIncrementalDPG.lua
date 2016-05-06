require 'torch'
require 'nn'

-- this class implement the Linear Deterministic Policy Gradient Method in Continuous Domain
-- Due to the constraint from "linear" and "continuous domain"
-- This is easier for us to put all the thing in this class


local LinearIncrementalDeterministicPolicy, parent = torch.class('rl.LinearDeterministicPolicy','rl.Incremental')



function LinearIncrementalDeterministicPolicy:__init(model, optimizer, criticOption, featureSize, gamma)
	self.model = model
	self.optimizer = optimizer
	
	self.gamma = gamma or 1
	
	-- for now we are only allow Q learning or Gradient Q learning for critic
	-- other options like batch method may be add in later
	if criticOption == "Q" then
		-- we are using a Linear Critic with Compatible Feature
		-- advantageCritic should have the input same to the row of the Jacobian matrix in compatible feature
		-- however, that row numer includes one bias number additional to the feature of the state
		-- thus, we have to add 1 for that bias
		self.advantageCritic = nn.Sequential(nn.Linear(featureSize + 1, 1))
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
	self.model:backward(s, self.desireAction)
	local gradient = self.optimizer.grads:clone()

	local asa = self.advantageCritic:forward(self:feature(s, self.action))
	local asprimea = self.advantageCritic:forward(self:feature(sprime, self.model:forward(sprime)))
	
	local vs = self.valueFunctionCritic:forward(s)
	local vsprime = self.valueFunctionCritic:forward(sprime)
	
	local qsa = asa + vs
	local qsprimea = asprimea + vsprime
	
	local tdError = r + self.gamma * qsprimea - qsa
	
	self.optimizer:gradientAscent(gradient)
	
	local advantageGradient = self:feature(s, self.action):mul(tdError)
	self.advantageParams:add(self.alr, advantageGradient)
	
	local valueFunctionGradient = s:clone():mul(tdError)
	self.valueFunctionParams:add(self.vlr, valueFunctionGradient)

end

-- return the compatible feature \phi(s,a)
function LinearIncrementalDeterministicPolicy:feature(s, a)
	local diff = a:clone():csub(self.desireAction)
	
	-- clear out the accumulated gradients and do a backward 
	self.optimizer.grads.zero()
	self.model:backward(s, self.desireAction)
	
	local gradient = self.optimizer.grads:clone()
	
	-- TODO : then reshape it to matrix form

	local compatibleFeature = torch.mv(gradient, diff)
	
	return compatibleFeature
end
