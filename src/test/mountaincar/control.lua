require 'rl'
require 'benchmark'


local mountainCar = nil

function init(randomStart)

	mountainCar = benchmark.MountainCar(randomStart)
	
	return mountainCar 
end





function step()
	-- do we need to restart the episode?
	if mountainCar.terminal then
		mountainCar:initialization()
	end
		
	local reward, state, terminal = mountainCar:step(1) -- for now, just a test action

		
		
end