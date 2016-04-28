require 'torch'


--[[
	This serves as the sampler for the function approximator to produce the continuous action to take.
	Specifically, this module is used to produce the action from an gaussian distribution
	See section 6 of paper : Simple statistical gradient-following algorithms for connectionist reinforcement learning
--]]

local GaussianPolicy, parent = torch.class('rl.GaussianPolicy','rl.Policy')


function GaussianPolicy:__init(actDim, stdev)
	parent:__init(actDim)
	-- if stdev is provided as a number, then we only adapt the mean
	-- if stdev is not provided, then we adapt it
	self.stdev = stdev
end

--[[
	g(a, mean, stdev) - Gaussian distribution function 
	a      - Sampled action from the distribution, a continuous value
	mean   - mean of the distribution
	stdev  - standard deviation of the distribution
	
	
	We are assuming action are not correlated
	each action has its own variance
	Input is the vector of [mean_1, ..., mean_i, stdev_1, ..., stdev_i], (if we are not adapting stdev, the vector become [mean_1, ..., mean_i])
--]]

function GaussianPolicy:forward(parameters)
	-- first test if we have enough parameters
	if self.stdev then
		assert(self.actNum == parameters:size()[1], 'mismatch of policy distribution')
	else
		assert(self.actNum * 2 == parameters:size()[1], 'mismatch of policy distribution')
	end
	
	-- save the input for backward computation
	self.input = parameters:clone()
		
	-- sample from the distribution for all the actions
	
	local stdevTable = nil
	local meanTable = nil
	
	if not self.stdev then
		-- if we are using an adaptive variance, separate it with means of actions
		stdevTable = parameters[{{self.actNum + 1, self.actNum + self.actNum}}]:clone():totable() -- get the distribution stdev
		meanTable = parameters[{{1, self.actNum}}]:clone():totable() -- get the distribution means
		
	else
		stdevTable = torch.Tensor(self.actNum):fill(self.stdev):totable() -- using fix stdev
		meanTable = parameters:clone():totable()
	end
	
	local actionTable = {}
	
	for i=1,#meanTable do
		actionTable[i] = torch.normal(meanTable[i], stdevTable[i])
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
function GaussianPolicy:backward()

	self.gradInput = self.input:clone()
	
	if self.stdev then
		-- we only backward the mean
		self.gradInput:csub(self.action):neg():div(self.stdev*self.stdev)
	else
		-- we have to adapt both

		local tempStdev = self.gradInput:narrow(1, self.actNum+1, self.actNum):clone()
		local tempMean = self.gradInput:narrow(1, 1, self.actNum):clone()
		
		-- first we do it with mean
		self.gradInput:narrow(1, 1, self.actNum):csub(self.action):neg():cdiv(tempStdev:clone():pow(2))	
		
		-- then we adapt stdev
		tempMean:csub(self.action):neg():pow(2):csub(tempStdev:clone():pow(2))
		tempMean:cdiv(tempStdev:clone():pow(3):add(0.000001))
		-- copy it back to the gradInput
		self.gradInput:narrow(1, self.actNum+1, self.actNum):copy(tempMean)
	end
			
	return self.gradInput

end