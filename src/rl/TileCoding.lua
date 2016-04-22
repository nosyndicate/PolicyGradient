
require 'torch'
require 'utils'

--[[
	This is an implementation of grid-style tile codings, heavily based on the c++ code from
	http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/steph/tiles.cpp
--]]

local TileCoding, parent = torch.class('rl.TileCoding','rl.Feature')


--[[
	numTiling specify the number of overlapping tile we want to use, it is recommended that it should be power of 2, e.g. 8
	scaleFactor used to specify the interval, for example, for a 2-dimensional space, 4*4 tilings will gridded the space into
	grid of size 0.25 * 0.25 (that is a unit grid (1*1 grid) divided into 4*4=16 smaller grids), if this parameter is not provided
	then we are assuming unit grid (1*1 tiling)
--]]
function TileCoding:__init(numTilings, scaleFactor, memorySize, safety)
	
	self.numTilings = numTilings
	self.scaleFactor = scaleFactor
	self.memorySize = memorySize
	
	self.maxLongInteger = 2147483647
	self.maxLongIntegerBy4 = math.floor(self.maxLongInteger/4)
	self.randomTable = {}
	self.collisionTable = utils.CollisionTable(memorySize, safety)
	
	
	for i = 1,2048 do
		-- the range are matching the original python code
		self.randomTable[i] = torch.random(0, self.maxLongIntegerBy4 - 1) 
	end
	
	self.quantizedState = {}
	self.offset = {}
	
end



-- this function compute the feature value given the state features
function TileCoding:feature(s)
	local f = s
	-- we first do scaling according to the scale factor
	-- see Example: 32 4x4 Rectangular Grid-Tilings of http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
	if not self.scaleFactor then
		assert(#s==#self.scaleFactor,"size of scale factor and feature do not match")
		for i=1,#s do
			f[i] = f[i]*self.scaleFactor[i]
		end
	end
	return self:tile(f)
end




function TileCoding:fixCoordination(numFeature, index)
	local coordinations = {}
	for i = 1, numFeature do
		if self.quantizedState[i] > self.offset[i] then
			coordinations[i] = self.quantizedState[i] - ((self.quantizedState[i] - self.offset[i]) % self.numTilings)
		else
			-- this equation comes from the python code by Sutton
			-- the document from University of New Hampshire mentioned that when quantizedState is less than the offset
			-- another equation is needed, however, the equation is not provided within the document
			-- see page 2 of http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tilesUNHdoc.pdf for details
			coordinations[i] = self.quantizedState[i] + 1 + ((self.offset[i] - self.quantizedState[i] - 1) % self.numTilings) - self.numTilings
		end
		-- compute displacement of next tiling in quantized space
		-- see page 4 of http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tilesUNHdoc.pdf for details
		self.offset[i] = self.offset[i] + 1 + (2*(i-1))
	end
	-- add additional indices for tiling and hashing_set so they hash differently
	coordinations[numFeature + 1] = index
	
	return coordinations
end

function TileCoding:tiles(feature)

	local numFeature = #feature
	local coordination = nil
			
	for i = 1, numFeature do
		self.offset[i] = 0
		-- quantize state to integers (henceforth, tile width == numTilings)
		self.quantizedState[i] = math.floor(feature[i] * self.numTilings)
	end
	
	local tilingList = {}
	
	for i = 1,numTiling do -- for each tiling
		coordination = self:fixCoordination(numFeature, i)
		if self.collisionTable then
			tilingList[i] = self:hash(coordination)
		else
			tilingList[i] = self:hashUNH(coordination, self.collisionTable.size)
		end
	end
	
	return tilingList
end

function TilingCoding:hash(integers)
	self.collisionTable.calls = self.collisionTable.calls + 1
	local memorySize = self.collisionTable.size
	
	local j = self:hashUNH(integers, #integers, memorySize)
	
	local collisionCheck = self:hashUNH(integers, self.maxLongInteger, 457)
		
	if collisionCheck == self.collisionTable.data[j] then  
		-- if new data same as saved data, add to hits
		self.collisionTable.clearHits = self.collisionTable.clearHits + 1
	elseif self.collisionTable.data[j] < 0 then
		-- first time, setup data
		self.collisionTable.clearHits = self.collisionTable.clearHits + 1
		self.collisionTable.data[j] = collisionCheck
	elseif self.collisionTable.safety == self.collisionTable.safetyLevel["unsafe"] then
		-- collision, but we don't care
		self.collisionTable.collisions = self.collisionTable.collisions + 1
	else -- handle collision, rehash
		local h2 = 1 + 2 * self:hashUNH(integers, self.maxLongIntegerBy4)
		local i = 1
		while collisionCheck ~= self.collisionTable.data[j] do -- keep looking for new spot, until we find an empty spot
			self.collisionTable.collisions = self.collisionTable.collisions + 1
			j = (j + h2) % memorysize
			
			-- test if we are running out of memory
			assert(i<= memorySize, "running out of memory")
			
			if self.collisionTable.data[j] < 0 then
				self.collisionTable.data[j] = collisionCheck
			end
			i = i + 1
		end
	end
end

function TileCoding:hashUNH(integers, size, increment)
	-- default value for increment in python code 449, no explanation
	increment = increment or 449
	local res = 0
	for i = 1, #integers do
		-- after mod 2048, the number is within range from 0 to 2047, thus we need to add 1
		-- to make the range right
		res = res + self.randomTable[(ints[i]+(i+1)*increment)%2048+1]
	end
	return res % size
end



