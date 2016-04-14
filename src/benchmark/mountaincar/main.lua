-- This is the graphical interface of the mountain car benchmark
-- This code uses the game engine love (https://love2d.org/)

-- add the needed module (rl, benchmark) to the package path
package.path = package.path..";../?/init.lua"
package.path = package.path..";../?.lua"

require 'torch'
require 'rl'
require 'benchmark'


-- global variables
local stepInteval = nil
local stepTimer = nil
local mountainCar = nil
local lower = nil
local upper = nil
local resolution = nil
local base = nil
local scale = nil

-- turn on th debugging
debug = true

function love.load(arg)
	-- create the game, do not use random start
	mountainCar = benchmark.MountainCar(false)
	stepInteval = 0.05 -- second
	stepTimer = stepInteval
	
	lower = mountainCar.minPosition
	upper = mountainCar.maxPosition
	
	base = 250
	scale = 120
		
	resolution = (upper - lower)/love.graphics.getWidth()
	
	-- set the background color to white
	love.graphics.setBackgroundColor(255,255,255)
end



function love.update(dt)
	 -- count the time loss
	stepTimer = stepTimer - dt
	
	-- time is up, need to take a step
	if stepTimer < 0 then
		
		-- do we need to restart the episode?
		if mountainCar.terminal then
			mountainCar:initialization()
		end
		
		mountainCar:step(1) -- for now, just a test action
		
		
		

		
		-- reset the timer
		stepTimer = stepInteval
	end
end

function mountain()
	local p = lower

	local x = 0;
	local y = -math.sin(3*p)*scale+base
	
	local points = {}
	local width = love.graphics.getWidth()

	for i=0,width-1 do
		x = i
		y = -math.sin(3*p) * scale + base
		p = p + resolution
		
		-- put the point into table
		table.insert(points, x)
		table.insert(points, y)
	end
	
	return points
end

function flag()
	local points = {}
	
	-- figure out the start point
	local goalPosition = mountainCar.goalPosition
	local x = (goalPosition - lower)/resolution
	local y = -math.sin(3*goalPosition)*scale+base
	
	table.insert(points,x)
	table.insert(points,y)
	
	table.insert(points,x)
	table.insert(points,y-20)
	
	table.insert(points,x+10)
	table.insert(points,y-20)
	
	table.insert(points,x)
	table.insert(points,y-10)
	
	return points
end


function car()
	local c = {}
	c.x = (mountainCar.position - lower) / resolution
	c.y = -math.sin(3*mountainCar.position) * scale + base
	
	return c
end

function love.draw()
	local p = lower

	local x = 0;
	local y = -math.sin(3 * p) * scale + base
	
	love.graphics.setColor(0,0,0)
	love.graphics.line(mountain())
	
	love.graphics.setColor(255,0,0)
	love.graphics.line(flag())
	
	
	local c = car()
	love.graphics.circle("fill", c.x, c.y, 5, 20)
	
	
end