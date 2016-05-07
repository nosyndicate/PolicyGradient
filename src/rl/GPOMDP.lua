require 'torch'


--[[
	See Table II in paper : Policy Gradient Methods for Robotics
	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/IROS2006-Peters_%5b0%5d.pdf
	for details
--]]

local GPOMDP, parent = torch.class('rl.GPOMDP','rl.PolicySearch')


function GPOMDP:__init(model, actor, optimizer, useOptimalBaseline)
	parent.__init(self, model, actor, optimizer)
	-- the default optimal baseline is turned off
	self.useOptimalBaseline = useOptimalBaseline or false
	self.rewardToCurrentStep = 0
	self.gradientEstimatorCurrentTrial = self.optimizer.grads:clone():zero()
	self.gradientToCurrentStep = self.optimizer.grads:clone():zero()
	self.trials = {}
	
end

function GPOMDP:startTrial()
	-- initialize the reward and gradient
	self.rewardToCurrentStep = 0
	self.gradientEstimatorCurrentTrial = self.optimizer.grads:clone():zero()
	self.gradientToCurrentStep = self.optimizer.grads:clone():zero()
end

function GPOMDP:endTrial()
	-- put the gradient estimator into the corresponding table for learning
	table.insert(self.trials, self.gradientEstimatorCurrentTrial)
end


function GPOMDP:step(s, r)

	-- clear out the gradient of previous steps
	-- NOTE: optimizer hold both the parameters and gradParameters
	self.optimizer.grads:zero()
	
	-- first accumulate the reward to current time step
	-- TODO: this means we are not discount the reward, may consider as future work
	self.rewardToCurrentStep = self.rewardToCurrentStep + r
	
	-- then compute the gradient of current step 
	local dLogPolicyDOutput = self.actor:backward()
		
	self.model:backward(s, dLogPolicyDOutput)
	
	self.gradientToCurrentStep:add(self.optimizer.grads)
	
	-- compute the gradient estimator from the beginning to current step
	local reward = torch.Tensor(self.gradientToCurrentStep:size()):fill(self.rewardToCurrentStep)
	local estimator = torch.cmul(self.gradientToCurrentStep, reward)
		
	-- add this estimator to the 
	self.gradientEstimatorCurrentTrial:add(estimator)
	
end



-- estimate the gradient use GPOMDP
function GPOMDP:calculateGradient(s, r)
	-- first get the number of trials
	local l = #self.trials
	-- create the gradient estimator
	local gradientEstimator = self.trials[1]:clone():zero()
	
	for i = 1,l do
		-- sum up the trials
		gradientEstimator:add(self.trials[i])
	end
	
	-- NOTE: need to reset the trials here
	self.trials = {}
	
	-- return the average of the gradient estimator
	return torch.div(gradientEstimator, l) 

end

-- TODO : have to implement the optimal baseline function
function GPOMDP:optimalBaseline()
	
end

