require 'torch'

local GPOMDP, parent = torch.class('rl.GPOMDP','rl.PolicySearch');


function GPOMDP:__init(model, actor, optimizer, useOptimalBaseline)
	parent.__init(self, model, actor, optimizer);
	-- the default optimal baseline is turned off
	self.useOptimalBaseline = useOptimalBaseline or false;
	self.rewardToCurrentStep = 0;
	self.gradient = optimizer.grads:clone():zero();
	self.gradientToCurrentStep = optimizer.grads:clone():zero();
	
end

-- estimate the gradient use GPOMDP
function GPOMDP:calculateGradient(s, r)
	-- clear the gradient
	self.optimizer.grads:zero();

	-- TODO: we are not discount the reward, may consider as future work
	self.rewardToCurrentStep = self.rewardToCurrentStep + r;
	local reward = torch.Tensor(self.gradientToCurrentStep:size()):fill(self.rewardToCurrentStep);
	
	-- then compute the gradient of current step
	local dLogPolicyDOutput = self.actor:backward();
	self.model:backward(s, dLogPolicyDOutput);

	-- accumulated the gradient
	self.gradientToCurrentStep:add(self.optimizer.grads);
	
	-- times the accumulated reward
	local tempGradient = torch.cmul(self.gradientToCurrentStep, reward);
	self.gradient:add(tempGradient);
	
	return self.gradient;
	
end

function GPOMDP:optimalBaseline()
	
end


