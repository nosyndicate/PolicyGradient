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
	
	-- first create the critic network and policy network
	if model then
		self.critic = args.critic
		self.policy = args.policy
	else
		self.critic = self:createCriticNetwork(args.batchNormalization)
		self.policy = self:createPolicyNetwork(args.batchNormalization)
	end

	-- and grab the parameter for the networks
	self.paramCritic, self.gradCritic = self.critic:getParameters()
	self.paramPolicy, self.gradPolicy = self.policy:getParameters()
	
	
	-- create the target networks
	if args.targets then
		self.targetCritic = model.critic:clone()
		self.targetPolicy = model.policy:clone()
		
		-- grab the parameters and gradient for target networks
		self.targetParamC, self.targetGradC = self.targetCritic:getParameters()
		self.targetParamP, self.targetGradP = self.targetPolicy:getParameters()
	else
		self.targetCritic = model.critic:clone()
		self.targetPolicy = model.policy:clone()
		
		-- NOTE:cannot call getParameter() on the target network, this will allocate new param and grad tensor
		self.targetParamC, self.targetGradC = self.paramCritic, self.gradCritic
		self.targetParamP, self.targetGradP = self.paramPolicy, self.gradPolicy
	end
	
	
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


function DDPG:createNetworks(batchNormalization)
	
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
	
	
	self.policy:train()
	self.critic:train()
	
	for i = 1,self.miniBatchPerIteratioin do
		local sSet, aSet, rSet, sprimeSet, tSet = self.transitionPool.sample(self.miniBatchSize)
		
		local criticEval = function(x)
			-- make sure we actually are optimizing the parameters
			assert(x==self.paramCritic)
		
			
		
		
		
			return 		
    	end
		-- optimize the critic 
		self.optim(criticEval ,self.paramCritic, self.optimState)
		
		
		
	end
	
end

