require 'torch'



local TransitionPool = torch.class('utils.TransitionPool')


--[[
This implements the replay memory buffer for Deep Reinforcement Learning
This code is heavily based on Google's code on Deep Q-network play Atari
However, this is used for continuous action, thus, the actNum change to actDim
--]]


function TransitionPool:__init(args)
	self.maxSize = args.maxSize
	self.numActor = args.numActor
	self.actionDim = args.actionDim
	self.stateDim = args.stateDim
	self.bufferSize = args.bufferSize
	self.numEntries = 0
    self.insertIndex = 0
    self.bufferIndex = -1
    
    
    -- we just use the default Tensor for storage for simplicity
    self.s = torch.Tensor(self.maxSize, self.stateDim):fill(0)
    self.sprime = torch.Tensor(self.maxSize, self.stateDim):fill(0)
   	-- this store the joint action value, for single agent, simply set self.numActor = 1
    self.a = torch.Tensor(self.maxSize, self.numActor*self.actionDim):fill(0)
    self.r = torch.Tensor(self.maxSize):fill(0)
    self.t = torch.Tensor(self.maxSize):fill(0)
    
    
    -- create batch buffer
    self.bufferS = torch.Tensor(self.bufferSize, self.stateDim):fill(0)
    self.bufferSprime = torch.Tensor(self.bufferSize, self.stateDim):fill(0)
    self.bufferA = torch.Tensor(self.bufferSize, self.numActor*self.actionDim):fill(0)
    self.bufferR = torch.Tensor(self.bufferSize):fill(0)
    self.bufferT = torch.Tensor(self.bufferSize):fill(0)
    
end


function TransitionPool:reset()
    self.numEntries = 0
    self.insertIndex = 0
end


function TransitionPool:size()
    return self.numEntries
end


function TransitionPool:empty()
    return self.numEntries == 0
end

function trans:fillBuffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers 
    -- use the random sample to fill in the buffer
    self.bufferIndex = 1
    for index=1,self.bufferSize do
        local s, a, sprime, r, t = self:sampleOne()
        self.bufferS[index] = s:clone()
    	self.bufferSprime[index] = sprime:clone()
    	self.bufferA[index] = a
    	self.bufferR[index] = r
    	self.bufferT[index] = t
    end
end

function TransitionPool:sample(batchSize)
	-- if the parameter is not set, use 1 as default
    local batchSize = batchSize or 1
    
    assert(batchSize < self.bufferSize)

	-- if the buffer is used out, fill in the new samples
    if self.bufferIndex == -1 or self.bufferIndex + batchSize - 1 > self.bufferSize then
        self:fillBuffer()
    end

    local batchStart = self.bufferIndex
    local batchEnd = self.bufferIndex + batchSize - 1
    local range = {{batchStart, batchEnd}}

	-- update buffer index
    self.bufferIndex = self.bufferIndex + batchSize

	return self.s[range], self.a[range], self.sprime[range], self.r[range], self.t[range]
end




function TransitionPool:sampleOne()
    assert(self.numEntries > 1)
    
    local index = torch.random(1, self.numEntries)
    
	return self.s[index], self.a[index], self.sprime[index], self.r[index], self.t[index]
end



-- s is tensor of state feature
-- a is the tensor of joint actions
function TransitionPool:add(s, a, sprime, r, t)
    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- increase until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone()
    self.a[self.insertIndex] = a:clone()
    self.r[self.insertIndex] = r
    
    
    if t then
    	-- if the sprime is terminal state, use the value from s instead
        self.t[self.insertIndex] = 1
        self.sprime[self.insertIndex] = s:clone()
    else
        self.t[self.insertIndex] = 0
        self.sprime[self.insertIndex] = sprime:clone()
    end
end

