require 'torch'

local StochasticGradientDescent, parent = torch.class('rl.StochasticGradientDescent','rl.Optimizer')


function StochasticGradientDescent:__init(params, grads)
	parent.__init(self, params, grads)
end

-- we are actually doing gradient ascent instead of descent
function StochasticGradientDescent:gradientAscent(gradient)
--	print("before")
--	print(self.params) 
	self.params:add(self.lr, gradient)
--	print("after")
--	print(self.params)
end