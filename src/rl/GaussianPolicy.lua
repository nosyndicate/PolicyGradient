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
	Input is the table of vectors {[mean_1, ..., mean_i]} {[stdev_1, ..., stdev_i]} (if we are not adapting stdev, the table is just {[mean_1, ..., mean_i]})
--]]

function GaussianPolicy:forward(parameters)
	-- first test if we have enough parameters
	assert(self.actNum == parameters[1]:size()[1], 'mismatch of policy mean')
	if not self.stdev then
		assert(self.actNum == parameters[2]:size()[1], 'mismatch of policy stdev')
	end
	

		
	-- sample from the distribution for all the actions
	local stdevTable = nil
	local meanTable = nil
	
	if not self.stdev then
		-- if we are using an adaptive variance, separate it with means of actions
		-- first save the input for backward computation
		self.input = {parameters[1]:clone(), parameters[2]:clone()}
		meanTable = parameters[1]:totable() -- get the distribution means
		stdevTable = parameters[2]:totable() -- get the distribution stdev
	else
		self.input = {parameters[1]:clone()}
		meanTable = parameters[1]:totable()
		stdevTable = torch.Tensor(self.actNum):fill(self.stdev):totable() -- using fix stdev
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

	
	if self.stdev then
		-- we only backward the mean
		self.gradInput = self.input:clone()
		self.gradInput:csub(self.action):neg():div(self.stdev*self.stdev)
	else
		-- we have to adapt both
		-- NOTE : here we return a table
		self.gradInput = {self.input[1]:clone(), self.input[2]:clone()}
		
		local temp = self.gradInput[1]:clone()
		
		-- first we do it with mean
		self.gradInput[1]:csub(self.action):neg():cdiv(self.input[2]:clone():pow(2))	
		
		-- then we adapt stdev
		temp:csub(self.action):neg():pow(2):csub(self.input[2]:clone():pow(2))
		temp:cdiv(self.input[2]:clone():pow(3):add(0.000001))
		-- copy it back to the gradInput
		self.gradInput[2]:copy(temp)
	end
			
	return self.gradInput

end