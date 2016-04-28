require 'rl'
require 'benchmark'
require 'nn'


local mountainCar = nil
local agent = nil
local trialCounter = 0
local state = nil
local trainingCounter = 0
local averageReward = 0



function init(randomStart)
	mountainCar = benchmark.MountainCar(randomStart)
	
	-- REINFORCE with neural net
	local model = nn.Sequential():add(nn.Linear(2, 3)):add(nn.Sigmoid()):add(nn.Linear(3,3)):add(nn.SoftMax())
	local policy = rl.SoftmaxPolicy(3)
	local optimizer = rl.StochasticGradientDescent(model:getParameters())
	agent = rl.Reinforce(model, policy, optimizer, true)
	agent:setLearningRate(0.01)
	
	
	return mountainCar 
end


function step(maximum)

	if mountainCar.terminal then
		state = torch.Tensor(mountainCar:reset())
		
		-- this is for the episodic method
		agent:startTrial()
	end

	local action = agent:getAction(state)
	
	
	local r, sprime, t = mountainCar:step(action-2) -- take a step 

	--print("action is "..action)

	agent:step(state, r)
	
	averageReward = averageReward + r

	-- go the the next state	
	state = torch.Tensor(sprime)
	
	-- for episodic method
	if t then
		agent:endTrial()
		trialCounter = trialCounter + 1
		
		-- only learn after 100 trajectory collected
		if trialCounter == 300 then
			agent:learn(nil, nil)
			print("learn once")
			
			trainingCounter = trainingCounter + 1
			trialCounter = 0
			
			print("average is ".. (averageReward/300.0))
			averageReward = 0
			
			-- training is finished
			if trainingCounter == maximum then
				return true
			end
		end
	end		
	
	-- training is not end
	return false
	
end


