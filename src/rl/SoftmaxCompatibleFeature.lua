require 'torch'

-- SoftmaxCompatibleFeature

local CompatibleFeature = torch.class('rl.CompatibleFeature');


function CompatibleFeature:__init(featureNum, actionNum)
	
end

-- this function compute the feature value given the state, action and policy
function CompatibleFeature:getFeatureValue(feature, action, pi)
	pi:forward()
end