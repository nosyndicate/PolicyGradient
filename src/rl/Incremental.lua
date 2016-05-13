local Incremental, parent = torch.class('rl.Incremental','rl.Learner')


function Incremental:__init(model, actNum)
	parent.__init(self, model, actNum)
end


function Incremental:setLearningRate(alpha)
	parent.setLearningRate(self, alpha)
end

function Incremental:getAction(s)
	
end



