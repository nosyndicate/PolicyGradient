require 'torch'

-- this is the parent class of SoftmaxCompatibleFeature and GaussianCompatibleFeature

local CompatibleFeature, parent = torch.class('rl.CompatibleFeature','rl.Feature');


function CompatibleFeature:__init()

end

-- this function compute the feature value given the state and action
function CompatibleFeature:feature(s)
	return s
end