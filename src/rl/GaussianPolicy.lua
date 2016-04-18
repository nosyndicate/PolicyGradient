require 'torch'


--[[
	This serves as the sampler for the neural network to produce the continuous action to take.
	Specifically, this module is used to produce the action from an gaussian distribution
	See section 6 of paper : Simple statistical gradient-following algorithms for connectionist reinforcement learning
--]]

local GaussianDistribution, parent = torch.class('rl.GaussianDistribution','rl.Policy')


function GaussianDistribution:__init(actDim, adaptiveVariance, fixedStdev)
	parent:__init(actDim)
	-- by default, we are only adapting mean of the gaussian distribution
	self.adaptiveVariance = adaptiveVariance or false
	self.stdev = fixedStdev or nil
end

--[[
	g(a, mean, stdev) - Gaussian distribution function 
	a      - Sampled action from the distribution, a continuous value
	mean   - mean of the distribution
	stdev  - standard deviation of the distribution
	
	
	We are assuming isotropic Gaussian function, which means the action are not correlated and with same variance
	Thus, \Sigma = \sigma^2 * I
	The covariance matrix is the same constant term times the identity matrix
	Input is the vector of [mean_1, ..., mean_i, stdev], (if we are not adapting variance, the vector become [mean_1, ..., mean_i]
--]]

function GaussianDistribution:forward(parameters)
	-- first test if we have enough parameters
	if not self.adaptiveVariance then
		assert(self.actNum == parameters:size()[1], 'mismatch of policy distribution')
	else
		assert(self.actNum + 1 == parameters:size()[1], 'mismatch of policy distribution')
	end
	
	-- save the input for backward computation
	self.input = parameters:clone()
		
	-- sample  from the distribution for all the actions
	
	local stdev = nil
	local meanTable = nil
	
	if self.adaptiveVariance then
		-- if we are using an adaptive variance, separate it with means of actions
		stdev = parameters[{o:size()[1]}] -- get the last element
		self.mean = parameters[{{1, o:size()[1]-1}}]:clone() -- get the distribution means
		meanTable = self.mean:totable() -- convert the distribution means to table
	else
		stdev = self.stdev -- using fix stdev
		self.mean = parameters:clone()
		meanTable = self.mean:totable()
	end
	
	local actionTable = {}
	
	for i=1,#meanTable do
		actionTable[i] = torch.normal(meanTable[i], stdev)
	end
	
	-- store the action in tensor for backward computation
	self.action = torch.Tensor(actionTable)
	
	
	-- return the action table
	return actionTable
end

--[[
	For Gaussian Policy, we only take the derivative with respect to the mean and stdev, 
	not with respect to the "real parameter"
	That is, we only compute 
	 d ln(g(a, mean, stdev))        (a - mean)              d ln(g(a, mean, stdev))       (a-mean)^2 - stdev^2
	-------------------------  =  --------------    and    -------------------------  =  -----------------------
	        d mean                    stdev^2                      d stdev                       stdev^3
	        
	if we assume the mean(s)  = \theta1^{T} * \phi(s) and stdev(s) = exp{\theta2^{T} * \phi(s)} 
	where \phi(s) is the feature of the states and \theta1 is the parameters of the mean and \theta2 is the parameters of stdev
	then we can get the gradient with respect to the "real parameter"
	for the mean, this term is 
	d ln(g(a, mean, stdev))         (a - mean)    
	-------------------------  =  -------------- * \phi(s)  
	        d \theta1                 stdev^2    
	        
	        
	for the stdev, the term is 
	 d ln(g(a, mean, stdev))       (a-mean)^2 - stdev^2
	-------------------------  =  ----------------------- * exp{\theta2^{T} * \phi(s)} * \phi(s)
	       d \theta2                      stdev^3
	                         
	                               (a-mean)^2 - stdev^2
	                           =  ----------------------- * \phi(s)
	                                      stdev^2
	                                      
	For the detail information of the later derivation, see paper : Model-Free Reinforcement Learning with Continuous Action in Practice
--]]
function GaussianDistribution:backward()

	self.gradInput = self.input:clone():zero()


	if self.adaptiveVariance then
		
	else
		
		


end