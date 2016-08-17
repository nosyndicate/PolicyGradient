require 'torch'
require 'nn'
require 'optim'




local DDPG, parent = torch.class('rl.DDPG','rl.Incremental')



function DDPG:__init(model, args)
	parent.__init(self, nil, args.actionDim)
	
	self.stateDim = args.stateDim
	self.actionDim = args.actionDim
	self.discountFactor = args.discountFactor or 0.99
	self.miniBatchSize = args.miniBatchSize 
	self.miniBatchPerIteration = args.miniBatchPerIteration or 5 -- this constant from the experiment section in continuous Q paper
	self.softTargetUpdate = args.softTargetUpdate or 0.001 -- this constant is from the Supplementary of DDPG paper
	-- default architecture value used in paper
	self.hiddenLayerUnits = args.hiddenLayerUnits or 200
	
	if args.batchNorm~=nil then
		self.batchNorm = true
	else
		self.batchNorm = args.batchNorm
	end
	
	-- first create the critic network and policy network
	if model then
		self.critic = model.critic
		self.policy = model.policy
	else
		self.critic = self:createCriticNetwork()
		self.policy = self:createPolicyNetwork()
	end

	-- and grab the parameter for the networks
	self.paramCritic, self.gradCritic = self.critic:getParameters()
	self.paramPolicy, self.gradPolicy = self.policy:getParameters()
	
	
	-- create the target networks
	self.targetCritic = model.critic:clone()
	self.targetPolicy = model.policy:clone()
		
	-- grab the parameters and gradient for target networks
	self.targetParamC, self.targetGradC = self.targetCritic:getParameters()
	self.targetParamP, self.targetGradP = self.targetPolicy:getParameters()

	-- then turn the target networks into evaluation mode
	-- target network are always in evaluate mode, thus batch normalization is disabled
	self.targetCritic:evaluate()
	self.targetPolicy:evaluate()
	
	
	-- initialize the Ornstein-Uhlenbeck process
	self.ouprocessArgs = {}
	self.ou = utils.OUProcess(ouprocessArgs)
	
	
	self.transitionPoolArgs = {}
	-- add in args for transition pool
	
	self.transitionPool = utils.TransitionPool(transitionPoolArgs)
	
	-- initialize optimization method
	self:initializeOptim(args.optim)
	
end


function DDPG:initializeOptim(args)
	self.optimState = args.optimState
	
	-- default we use adam
	if not args.optim then
		self.optimMethod = optim.adam
		return
	end
	
	-- choose according to the args
	if args.optim == 'sgd' then
		self.optimMethod = optim.sgd
	elseif args.optim == 'adam' then
		self.optimMethod = optim.adam
	end
end

function DDPG:createCriticNetwork(batchNorm)
	-- this configuration of hyperparameter came from deep continuous Q learning
	local network = nn.Sequential()

	network:add(nn.Linear(self.stateDim + self.actionDim, self.hiddenLayerUnits))
	if self.batchNorm then
		network:add(nn.BatchNormalization(self.hiddenLayerUnits))
	end
	network:add(nn.ReLU())
	
	network:add(nn.Linear(self.hiddenLayerUnits, self.hiddenLayerUnits))
	if self.batchNorm then
		network:add(nn.BatchNormalization(self.hiddenLayerUnits))
	end
	network:add(nn.ReLU())
	
	network:add(nn.Linear(self.hiddenLayerUnits, 1))
	return network
end


function DDPG:createPolicyNetwork(batchNorm)
	local network = nn.Sequential()

	network:add(nn.Linear(self.stateDim, self.hiddenLayerUnits))
	if self.batchNorm then
		network:add(nn.BatchNormalization(self.hiddenLayerUnits))
	end
	network:add(nn.ReLU())
	
	network:add(nn.Linear(self.hiddenLayerUnits, self.hiddenLayerUnits))
	if self.batchNorm then
		network:add(nn.BatchNormalization(self.hiddenLayerUnits))
	end
	network:add(nn.ReLU())
	
	network:add(nn.Linear(self.hiddenLayerUnits, self.actionDim))
	return network
end

function DDPG:setAdditionalLearningRate(alr, vlr)
	self.alr = alr
	self.vlr = vlr
end

function DDPG:initiateParameters(lower, upper)
	self.optimizer.params:uniform(lower, upper)
end

function DDPG:getAction(s)
	-- not training, sample an action from policy
	-- first disable switch to test mode if batch normalization is in use
	self.policy:evaluate()
	self.critic:evaluate() -- this is probably unnecessary, but we put it here anyway

	self.action = self.policy:forward(s)
	-- add some noise from the OU process
	
	self.action = self.ou:getAction(self.action)
	
	return self.action:totable()
end



function DDPG:learn(s, r, sprime, terminal)
	-- first store the transition (s, a, r, sprime) into the memory
	-- note the format of each element in tuple. s, a, sprime are tensors, r is a scalar
	
	self.transitionPool:add(s, self.action, r, sprime, terminal)

	for i = 1,self.miniBatchPerIteratioin do
		local sSet, aSet, rSet, sprimeSet, tSet = self.transitionPool.sample(self.miniBatchSize)
		
		
		-- optimize the critic 
		-- first get into the training mode for the critic
		self.critic:train()
		local criticEval = function(x)
			-- make sure we actually are optimizing the critic's parameters
			assert(x==self.paramCritic)
		
			-- first compute the y value by target networks
			local aprime = self.targetPolicy:forward(sprimeSet)
			-- then we put the sprime align with aprime to construct the input for target critic network
			local targetInput = torch.cat(sprimeSet, aprime, 2)
			
			local targetQVal = self.targetCritic:forward(targetInput)
			
			-- compute y = r + (1-terminal) * gamma * qVal
			local y = tSet:clone():mul(-1):add(1)
			y:mul(self.discountFactor):cmul(targetQVal):add(rSet:clone())
						
			-- input for the critic
			local input = torch.cat(sSet, aSet, 2)			
			
			-- get the q value for state s, action a
			local qVal = self.critic:forward(input)
			
			-- since we are minimizing the square loss, we compute the dLoss/dQ
			-- NOTE, we are doing gradient descent (not ascent) on loss with method from optim package 
			-- thus, we need to multiply -1 at the end which come from the coefficient of the critic (not target critic)
			-- see section 3.1.2 from "A brief survey of parametric value function approximation"
			local dLossdQ = y:add(-1, qVal):mul(-1)
			
			-- zero the gradient of the critic network
			self.gradCritic:zero()
			
			-- get the gradient of the Loss w.r.t parameters of Q network
			self.critic:backward(input, dLossdQ)
			
			-- the Loss funciton L is the average over all samples in the mini batch
			-- so we have do divide the number of samples in mini batch
			-- same with the loss function
			self.gradCritic:div(self.miniBatchSize)
			
			-- compute the loss = f(x)
			local loss = dLossdQ:clone():pow(2):div(self.miniBatchSize):cumsum()
	
		
			return loss, self.gradCritic
    	end
    	
		self.optim(criticEval ,self.paramCritic, self.optimState)
		
		
		-- optimize the policy
		-- first get into the training mode for the policy
		-- and disable the training mode for critic
		self.critic:evaluate()
		self.policy:train()
		local policyEval = function(x)
    		-- make sure we actually are optimizing the policy's parameters
			assert(x==self.paramPolicy)
			
			-- first get the action from the policy
			local actions = self.policy:forward(sSet)
			
			-- construct the input for critic, same as before
			local input = torch.cat(sSet, actions, 2)
			
			-- first get Q
			self.critic:forward(input)
			
			-- zero the gradient of the poicy and critic network
			self.gradPolicy:zero()
			self.gradCritic:zero() -- grad for critic probably do not need to be zero out
			
			-- we want to know the gradient of Q with respect to the input, especially for the action part
			local gradOutput = torch.Tensor(self.miniBatchSize, 1):fill(1)
			local gradInput = self.critic:backward(input, gradOutput)
			
			-- get action part from gradInput
			local gradAction = gradInput:narrow(2, self.stateDim+1, self.actionDim):clone()
			-- then pass it to the policy
			self.policy:backward(sSet, gradAction)
			
			-- divided by the mini batch size get the average gradient like in the critic part
			-- unlike critic part, we are doing gradient ascent, so we times -1
			self.gradPolicy:div(self.miniBatchSize):mul(-1)
			
			-- we do not care about the f(x) here, so just ignore it
			return 0, self.gradPolicy
    	end
		
		self.optim(criticEval ,self.paramPolicy, self.optimState)
		
		
		-- soft update of the target networks
		self.targetParamC:mul(1-self.softTargetUpdate):add(self.softTargetUpdate, self.paramCritic)
		self.targetParamP:mul(1-self.softTargetUpdate):add(self.softTargetUpdate, self.paramPolicy)
		
		
	end
	
end

