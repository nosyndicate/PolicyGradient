require 'torch'

local OUProcess = torch.class('utils.OUProcess')

--[[
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    dWt = W_t - W_{t-1} ~ N(0,1)
--]]


function OUProcess:__init(args)
	self.mu = args.mu or 0
	self.theta = args.theta or 0.15
	self.sigma = args.siga or 0.3
	
	self:reset()
end


function OUProcess:reset()
	self.state = torch.ones(self.actionDim):mul()
end


function OUProcess:evolveState()
	local x = self.state
	local dx = self.theta * (self.mu - x) + self.sigma * torch.
	self.state = x + dx
	return self.state
	
end

function OUProcess:getAction(policy, state)
	local action = policy:getAction(state)
	local ouState = self:evolveState()
	return 
end
