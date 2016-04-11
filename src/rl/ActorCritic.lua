require 'torch'

-- Implementation of incremental actor critic, using Sarsa critic
-- based on policy gradient lecture of David Silver 
-- http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf
-- and nips paper "Incremental Natural Actor-Critic Algorithms"
-- https://papers.nips.cc/paper/3258-incremental-natural-actor-critic-algorithms.pdf

local ActorCritic, parent = torch.class('rl.Reinforce','rl.PolicySearch');


function ActorCritic:__init(model, actor, critic, optimizer)
	-- parent method have to be called this way: with dot and pass self as first parameters
	parent.__init(self, model);
	self.actor = actor;
	
	self.optimizer = optimizer;
	--self.optimizer.params:uniform(-0.08,0.08);
end

function ActorCritic:setLearningRate(alpha)
	parent.setLearningRate(self, alpha);
	self.optimizer:setLearningRate(self.alpha);
end

function ActorCritic:step(s, r)

end

function ActorCritic:getAction(s)
	-- get the parameters for the distribution of the stochastic policy
	local parameters = self.model:forward(s);
	--print(parameters);
	-- sample from the distribution 
	local action = self.actor:getAction(parameters);
	return action;
end

function ActorCritic:learn(s, r)
	local gradient = self:calculateGradient(s, r);
	
	-- update the parameters with the gradient
	self.optimizer:gradientAscent(gradient);
end

function ActorCritic:calculateGradient(s, r)
	
end






