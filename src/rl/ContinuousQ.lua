require 'torch'
require 'utils'

local ContinuousQ, parent = torch.class('rl.ContinuousQ','rl.Incremental')

--[[


--]]
function ContinuousQ:__init(model, args)
	parent.__init(self, model, actNum)
	
	-- determine if we need to create a default networks 
	if model then
		self.vNetwork = model.vNetwork
		self.meanNetwork = model.meanNetwork
		self.covarianceNetwork = model.covarianceNetwork
	else
		self.vNetwork, self.meanNetwork, self.covarianceNetwork = model.createNetworks()
	end
	
	-- grad the parameters and gradient from each network
	self.paramV, self.gradV = self.vNetwork:getParameters()
	self.paramM, self.gradM = self.meanNetwork:getParameters()
	self.paramC, self.gradC = self.covarianceNetwork:getParameters()
	
	-- determine if we are using target network
	if args.targetQ then
		self.targetVNetwork = model.vNetwork:clone()
		self.targetMeanNetwork = model.meanNetwork:clone()
		self.targetCovarianceNetwork = model.covarianceNetwork:clone()
		
		-- grab the parameters and gradient for target networks
		self.targetParamV, self.targetGradV = self.targetVNetwork:getParameters()
		self.targetParamM, self.targetGradM = self.targetMeanNetwork:getParameters()
		self.targetParamC, self.targetGradC = self.targetCovarianceNetwork:getParameters()
	else
		self.targetVNetwork = model.vNetwork
		self.targetMeanNetwork = model.meanNetwork
		self.targetCovarianceNetwork = model.covarianceNetwork
		
		-- NOTE:cannot call getParameter() on the target network, this will allocate new param and grad tensor
		self.targetParamV, self.targetGradV = self.paramV, self.gradV
		self.targetParamM, self.targetGradM = self.paramM, self.gradM
		self.targetParamC, self.targetGradC = self.paramC, self.gradC
	end
	
	self.stateDim = args.stateDim
	self.actionDim = args.actionDim
	self.discountFactor = args.discountFactor or 0.99
	self.miniBatchSize = args.miniBatchSize 
	self.miniBatchPerIteration = args.miniBatchPerIteration or 5 -- this constant from the experiment section in continuous Q paper
	self.softTargetUpdate = args.softTargetUpdate or 0.001 -- this constant is from the Supplementary of DDPG paper
	
	
	
	-- default architecture value used in paper
	self.hiddenLayerUnits = args.hiddenLayerUnits or 200
	
	self.transitionPoolArgs = {}
	-- add in args for transition pool
	
	self.transitionPool = utils.TransitionPool(transitionPoolArgs)
end



function ContinuousQ:setLearningRate(alpha)
	parent.setLearningRate(self, alpha)
end

-- action is based on the Gaussian distribution with the \mu and \Sigma from the Quadratic form
function ContinuousQ:getAction(s)
 
end


-- direct update the policy or value function
function ContinuousQ:learn(s, a, r, sprime, terminal)
	-- first store the transition (s, u, r, sprime) into the memory
	-- note the format of each element in tuple. s, r, sprime are tensors, r is a scalar
	
	self.transitionPool:add(s, a, r, sprime, terminal)
	
	for i = 1,self.miniBatchPerIteratioin do
		local sSet, aSet, rSet, sprimeSet, tSet = self.transitionPool.sample(self.miniBatchSize)
		
		local y = torch.Tensor(self.miniBatchSize)
		
		-- compute V'(s_{t+1})
		local v = self.targetNetwork():forward(sprimeSet)[1]
		
		-- delta = r + (1 - t) * gamma * V' - Q
		local term = tSet:clone():mul(-1):add(1)
		local delta = v:clone():mul(self.discountFactor):cmul(term)
		delta:add(rSet)
		
		-- TODO : forward method to compute the mean, covariance and v
		-- then use those to compute Q
		
		
		-- if the loss function is L, we first compute 
		-- dL/dQ = -delta
		-- however, we will doing minimization so \theta = \theta - dL/dQ
		-- thus, we ignore the negative sign here, and update the weight by adding dL/dQ



		-- we update the target network with softTargetUpdate
		-- break this into two step, since the target network could be the original network
		-- also vChange is a new tensor, since we don't want to change the value in paramV
		-- this is same with the remain network
		local vChange = torch.mul(self.paramV, self.softTargetUpdate)
		self.targetParamV:mul(1-self.softTargetUpdate):add(vChange)
		
		local mChange = torch.mul(self.paramM, self.softTargetUpdate)
		self.targetParamM:mul(1-self.softTargetUpdate):add(mChange)
		
		local cChange = torch.mul(self.paramC, self.softTargetUpdate)
		self.targetParamC:mul(1-self.softTargetUpdate):add(cChange)

	end
		
	
	
	
end

--[[
	ActionDim=du, Statedim=dx

	Architectures primarily used for the experimental results reported on 
	the paper are consist of three DISJOINT networks taking in state as 
	input, outputting:
		mu (dims=du): mu in the paper
		L  (du*(du+1)/2) : lower triangular matrix
		V   (1): V in the paper

	As final output activation, tanh is applied to mu output. And exp() is 
	applied to diagonals of L. Others are linear.

	In terms of architecture we used two hidden layers and searched through 
	some hyperparameters as they are on the paper.
--]]

function ContinuousQ:createNetwork()
	-- continuous Q network have three parts
		
	local valueThread = nn.Sequential()
	valueThread:add(nn.Linear(self.stateDim, self.hiddenLayerUnits)):add(nn.ReLU())
	valueThread:add(nn.Linear(self.hiddenLayerUnits, self.hiddenLayerUnits)):add(nn.ReLU())
	valueThread:add(nn.Linear(self.hiddenLayerUnits, 1))
	
	local meanThread = nn.Sequential()
	meanThread:add(nn.Linear(self.stateDim, self.hiddenLayerUnits)):add(nn.ReLU())
	meanThread:add(nn.Linear(self.hiddenLayerUnits, self.hiddenLayerUnits)):add(nn.ReLU())
	meanThread:add(nn.Linear(self.hiddenLayerUnits, self.actionDim)):add(nn.Tanh())
	
	local covarianceThread = nn.Sequential()
	covarianceThread:add(nn.Linear(self.stateDim, self.hiddenLayerUnits)):add(nn.ReLU())
	covarianceThread:add(nn.Linear(self.hiddenLayerUnits, self.hiddenLayerUnits)):add(nn.ReLU())

	--TODO: finish the architecture here
	
end


