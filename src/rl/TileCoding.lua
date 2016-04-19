require 'torch'

-- this is the parent class of SoftmaxCompatibleFeature and GaussianCompatibleFeature

local TileCoding, parent = torch.class('rl.TileCoding','rl.Feature');


function CompatibleFeature:__init()

end

-- this function compute the feature value given the state and action
function CompatibleFeature:feature(s)
	return s
end