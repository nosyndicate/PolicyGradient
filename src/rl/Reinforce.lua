require 'torch'

local Reinforce, parent = torch.class('rl.Reinforce','rl.PolicySearch')


function Reinforce:__init(model, actor, optimizer, useOptimalBaseline)
	parent.__init(self, model, actor, optimizer)
	--self.reward = {}
	--self.gradient = {}
	self.trials = {}
	self.rewardCurrentTrial = nil
	self.gradientCurrentTrial = nil
	-- the default optimal baseline is turned off
	self.useOptimalBaseline = useOptimalBaseline or false
end

function Reinforce:step(s, r)

	-- clear out the gradient of previous steps
	-- NOTE: optimizer hold both the parameters and gradParameters
	self.optimizer.grads:zero()
	
	-- first accumulate the reward
	-- TODO: this means we are not discount the reward, may consider as future work
	self.rewardCurrentTrial = self.rewardCurrentTrial + r
	
	-- then compute the gradient of current step and add to gradient of current trial
	local dLogPolicyDOutput = self.actor:backward()

	self.model:backward(s, dLogPolicyDOutput)
	print("state")
	print(s)
	
	print(self.optimizer.grads)
	print("============")
	self.gradientCurrentTrial:add(self.optimizer.grads)
end

function Reinforce:startTrial()
	-- initialize the reward and gradient
	self.rewardCurrentTrial = 0
	self.gradientCurrentTrial = self.optimizer.grads:clone():zero()
end

function Reinforce:endTrial()
	-- put the reward and gradient into the corresponding table for learning
	table.insert(self.trials,{gradient = self.gradientCurrentTrial, reward = self.rewardCurrentTrial})
end


-- episodic Reinforce only learns at the end of trials
function Reinforce:calculateGradient(s, r)
	-- first get the number of trials
	local l = #self.trials
	-- create the gradient estimator
	local gradientEstimator = self.trials[1].gradient:clone():zero()
	
	-- default optimal baseline is 0
	local optimalBL = gradientEstimator:clone():zero()
	if self.useOptimalBaseline then
		optimalBL = self:optimalBaseline()
	end
	
	for i = 1,l do
		local reward = torch.Tensor(gradientEstimator:size()):fill(self.trials[i].reward)
		if self.useOptimalBaseline then
			reward:csub(optimalBL)
		end
		--print("trial "..i)
		--print(self.trials[i].gradient)
		-- multiple the accumulated gradient for each trial with the reward
		local tempGradient = torch.cmul(self.trials[i].gradient, reward)
		-- sum up the trials
		gradientEstimator:add(tempGradient)
	end
	
	-- NOTE: need to reset the trials here
	self.trials = {}
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l) 
end

function Reinforce:optimalBaseline()
	local l = #self.trials
	local numerator = self.trials[1].gradient:clone():zero()
	local denominator = numerator:clone()
	for i = 1,l do
		local innerProduct = torch.cmul(self.trials[i].gradient, self.trials[i].gradient)
		numerator:add(torch.mul(innerProduct, self.trials[i].reward))
		denominator:add(innerProduct)
	end
	local optimalBL = torch.cdiv(numerator, denominator)
	
	return optimalBL
end


