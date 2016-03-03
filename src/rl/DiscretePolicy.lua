require 'torch'


--[[
	This serves as the sampler for the neural network to produce the discrete action to take.
	Specifically, this module is used to produce the multinomial action in one hot encoding e.g. (0 0 0 1 0 ... 0)
	See section 6 of paper : Simple statistical gradient-following algorithms for connectionist reinforcement learning
	This code heavily use the code from dpnn : https://github.com/nicholas-leonard/dpnn/blob/master/ReinforceCategorical.lua
--]]

local DiscretePolicy, parent = torch.class('rl.DiscretePolicy','rl.Policy');


function DiscretePolicy:__init(actNum)
	parent:__init(actNum);
end

--[[
	g(y,p_1,p_2,p_3,...,p_n) - Multinomial distribution function 
	y      - Sampled values (1 to n) (self.action will encoding the sampled value in one-hot format)
	p_i    - Probability of sampling a value of i
	g(y,p_1,p_2,p_3,...,p_n) = p_i   (if y = i)
	       
	Input is the vector of p_i (p_1, p_2, p_3, ..., p_n)
	The sampled value from the Multinomial distribution in one-hot encoding will be store in self.action
	The output of this function is the index of the action
--]]
function DiscretePolicy:forward(parameters)
	assert(self.actNum == parameters:size()[1], 'mismatch of policy distribution');
	
	-- save the input for future use
	self.input = parameters:clone();
	
	-- sample 1 time without replacement from the distribution
	local index = torch.multinomial(parameters, 1); 
	
	-- convert the index into one-hot representation
	self.action = parameters:clone();
	-- set all element to zero
	self.action:zero();
	-- put 1 at the index along the first dimension (row)
	-- encode in one-hot encoding
	self.action:scatter(1, index, 1);
	
	-- return the index in number	
	return self.action:nonzero()[1][1];
end

--[[
	We measure the derivative of log Multinomial w.r.t p
	p - probability vector (p_1, p_2, p_3, ..., p_n)
	  d ln(g(y,p))           1            d g(y,p)
	---------------- =  -----------  *  ------------
	      d p              g(y,p)           d p
	
	The derivative is a vector, where at position i
	=    1/p_i * 1          (if y = i)
	=    0                  (if y = 0)
--]]
function DiscretePolicy:backward()
	self.gradInput = self.action:clone();
	local denominator = self.input:clone();
	
	-- add some tiny value in case some probability is 0
	-- denominator = self.input:clone():add(0.00000001);
	
	self.gradInput:cdiv(denominator);
	
	return self.gradInput;
end