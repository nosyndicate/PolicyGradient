require 'torch'

-- this is the parent class of all feature

local Feature = torch.class('rl.Feature');


function Feature:__init()
	
end

-- this function compute the feature value given the state and action
function Feature:feature(s)
	return s
end