require 'torch'

-- this is the parent class of SoftmaxCompatibleFeature and GaussianCompatibleFeature

local CompatibleFeature = torch.class('rl.CompatibleFeature');


function CompatibleFeature:__init()

end

-- this function compute the feature value given the state and action
function CompatibleFeature:getFeatureValue()

end