require 'torch'


--[[
	The abstract class of policy
--]]

local Policy = torch.class('rl.Policy')


function Policy:__init(actNum)
	-- in discrete cases, we use actNum for action number
	-- in continuous case, we use actNum for action dimension
	self.actNum = actNum
	self.action = torch.Tensor()
	self.input = torch.Tensor()
	self.gradInput = torch.Tensor()
end



-- simply calls the forward function
function Policy:getAction(parameters)
	return self:forward(parameters)
end

-- receive the parameter (which is the output of the model) and return an action
function Policy:forward(parameters)

end

-- compute the derivative of the action with respect to the parameters
function Policy:backward()

end