require 'torch'

local Learner = torch.class('rl.Learner')

function Learner:__init(model, actNum)
	self.model = model
	self.actNum = actNum
	self.alpha = 0.001
end

function Learner:setLearningRate(alpha)
	self.alpha = alpha
end

-- receive the state as tensor, return the action
-- the return type is a index of the action (discrete case)
-- or table of continuous values
function Learner:getAction(s)
	
end

-- do some work before each episode start
function Learner:startTrial()

end

-- do some work after the episode ends
function Learner:endTrial()
end

-- step function, do something after each step for learning
-- thus, this function should not update any weights for policy or value for value function
function Learner:step(s, r)

end

-- direct update the policy or value function
function Learner:learn(s, r, sprime)
	
end

