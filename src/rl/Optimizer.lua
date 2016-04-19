require 'torch'

local Optimizer = torch.class('rl.Optimizer')

function Optimizer:__init(params, grads)
	self.params = params
	self.grads = grads
	self.lr = 0.001
end

function Optimizer:setLearningRate(alpha)
	self.lr = alpha
end

-- we are actually doing gradient ascent instead of descent
function Optimizer:gradientAescent()

end