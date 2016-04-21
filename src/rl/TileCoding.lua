
require 'torch'
require 'utils'

--[[
	This is an implementation of grid-style tile codings, heavily based on the python code from
	http://rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
--]]

local TileCoding, parent = torch.class('rl.TileCoding','rl.Feature')


function TileCoding:__init()
	self.maxNumFloats = 20
	self.maxLongInteger = 2147483647
	self.maxLongIntegerBy4 = math.floor(self.maxLongInteger/4)
	self.randomTable = {}
	
	for i = 1,2048 do
		self.randomTable[i] = 
	end
	
	self.qstate = {}
	self.base = {}
	for i = 1,self.maxNumFloats do
		self.qstate[i] = 0
		self.base[i] = 0
	end
	
end



-- this function compute the feature value given the state and action
function TileCoding:feature(s)
	return s
end

-- Does initial assignments to coordinates
function startTiles(coordinations, numTilings, floats, ints)
	local numFloats = #floats
	local i = numFloats + 1
	-- first place the integer value
	for i = 1,#ints do
		coordinations[offset + i] = int
end

function tiles(numTilings, floats, ints)
	ints = ints or {} -- if ints is not used, initialize to empty table
	local numFloats = #floats
	local numCoordination = 1 + numFloats + #ints
	local coordination = {}
	
	self:startTiles(coordination, numTilings, floats, ints)
end



