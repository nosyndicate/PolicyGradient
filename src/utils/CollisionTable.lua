require 'torch'

local CollisionTable = torch.class('utils.CollisionTable')



-- This handles the collision in hashing for Tile Coding
-- safety has two options : unsafe, safe
-- 

--[[
	the two parameters are optional, however, size must be a power of 2
	safety level has three options:
	1. unsafe: note the number of collisions, but don't handle them
	2. safe: keeps an extra hash numbers to check for collisions; it is possible (although unlikely) that a collision will be 
	determined to be a simple hit instead
--]]
function CollisionTable:__init(size, safety)
	
	self.safetyLevel = {["unsafe"]=0,["safe"]=1}
	
	self.size = size or 2048 -- default size
	self.safety = safetyLevel[safety]
	
	assert(powerOf2(self.size), 'size should be power of 2')
	
	self.calls = 0
	self.clearHits = 0
	self.collisions = 0
	self.data = {}
	for i = 1,self.size do
		table.insert(self.data, -1)
	end
		
end


function CollisionTable:info()
	local infoTable = {"Collision table: ", "Safety : ", self.safety
    	" Usage : ", self.usage()
		" Size :", self.size
		" Calls : ", self.calls
		" Collisions : ", self.collisions}

	-- use concat for concatenating many short sentence
	return table.concat(infoTable)
end

function CollisionTable:reset()
	self.calls = 0
	self.clearHits = 0
	self.collisions = 0
	self.data = {}
	for i = 1,self.size do
		table.insert(self.data, -1)
	end
end

function CollisionTable:stats()
	return {self.calls, self.clearHits, self.collisions, self.usage}
end

function CollisionTable:usage()
	local use = 0
	for i = 1,#data do
		if data[i] >= 0 then
			use = use + 1
		end
	end
	return use
end

function powerOf2(size)
	local lgSize = math.log10(size)/math.log10(2) -- changing the base to 2
	return (lgSize - math.floor(lgSize)) == 0
end

function mod(number, by)
	if number >= 0 then
		return num % by
	else
		return (by + (num % by)) % by
	end
end