local Incremental, parent = torch.class('rl.GQLambda','rl.Incremental')


function GQLambda:__init(actNum, useEligibilityTrace)
	parent.__init(actNum)
	
	

	-- by default, we are not using eligibility trace
	self.useEligibilityTrace = useEligibilityTrace or false
	
	
end


function GQLambda:setLearningRate(alpha)
	parent.setLearningRate(self, alpha)
	
	-- 
	
end

function GQLambda:getAction(s)
	
end


function Learner:learn(s, r)
	
end
