require 'torch'



local MountainCar = torch.class('benchmark.MountainCar');

-- Constructor
function MountainCar:__init(randomStart)
	self.randomStart = randomStart
	
	-- 0 is within valid range
	self.position = 0
	self.velocity = 0
	
	-- other variables
	self.minPosition = -1.2
	self.maxPosition = 0.6
	self.maxVelocity = 0.07
	self.goalPosition = 0.5
	self.terminal = false
end


-- set the position and velocity of the robot
function MountainCar:initialization()

	if not self.randomStart then
		self.position = -0.5
		self.velocity = 0
	else
		-- note, the uniform function is sample from [a,b), not [a,b]
		self.position = torch.uniform(self.minPosition,self.maxPosition)
		self.velocity = torch.uniform(-self.maxVelocity,self.maxVelocity)
	end
	
	self.terminal = false
end


-- Move the car
-- action is in set {-1,0,1}
function MountainCar:step(action)
	local reward = -1;


	local temp = self.velocity + 0.001 * action - 0.0025 * math.cos(3*self.position)
	self.velocity = math.max(math.min(temp, self.maxVelocity), -self.maxVelocity)
	self.position = self.position + self.velocity;
	self.position = math.max(math.min(self.position, self.maxPosition), self.minPosition)
	
	if self.position==self.minPosition and self.velocity < 0 then
		self.velocity = 0
	end
	
	if self.position >= self.goalPosition then
		self.terminal = true
	end
	
	return reward, {self.position, self.velocity}, self.terminal
end


