require 'torch'

-- this is the parent class of all feature

local InvertingGradient = torch.class('rl.InvertingGradient');


function InvertingGradient:__init()

end

-- this function compute the feature value given the state and action
function InvertingGradient:boundGradient(pgradient, pmin, pmax, p)
  if pgradient > 0 then -- if gradient suggests increasing p
    return pgradient * (pmax-p)/(pmax - pmin)
  end
  return pgradient * (p - pmin)/(pmax - pmin)
end
