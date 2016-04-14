require 'torch'


--[[
	This serves as the sampler for the neural network to produce the continuous action to take.
	Specifically, this module is used to produce the action from an gaussian distribution
	See section 6 of paper : Simple statistical gradient-following algorithms for connectionist reinforcement learning
--]]

local GaussianDistribution, parent = torch.class('rl.GaussianDistribution','rl.Policy');


function GaussianDistribution:__init(actDim, adaptiveVariance)
	parent:__init(actDim);
	-- by default, we are only adapting mean of the gaussian distribution
	self.adaptiveVariance = adaptiveVariance or false;
	self.stdev = nil;
end

--[[
	g(a, mean, stdev) - Gaussian distribution function 
	y      - Sampled action from the distribution, a continuous value
	mean   - mean of the distribution
	stdev  - standard deviation of the distribution
	
	
	We assuming isotropic Gaussian function, which means the 
	\Sigma = \sigma^2 * I
	The covariance matrix is the same constant term times the identity matrix, actions are not correlated
	Input is the vector of [mean_1, ..., mean_i, variance], (if we are not adapting variance, the vector become [mean_1, ..., mean_i]
--]]

function GaussianDistribution:forward(parameters)
	if not self.adaptiveVariance then
		assert(self.actNum == parameters:size()[1], 'mismatch of policy distribution');
	else
		assert(self.actNum + 1 == parameters:size()[1], 'mismatch of policy distribution');
	end
	
	-- save the input for future use
	self.input = parameters:clone();
	
	
	
	
	-- sample 1 time without replacement from the distribution
	-- for all the actions

	
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


function GaussianDistribution:backward()
	self.gradInput = self.action:clone();
	local denominator = self.input:clone();
	
	-- add some tiny value in case some probability is 0
	-- denominator = self.input:clone():add(0.00000001);
	
	self.gradInput:cdiv(denominator);
	
	return self.gradInput;
end