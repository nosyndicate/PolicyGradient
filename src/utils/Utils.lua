

function utils.callFunctionOnObjects(functionName, objectList, parametersList)

	local resultList = {}

	if parametersList then
		if #parametersList ~= 1 then
			assert(#objectList == #parametersList, "mismatch of the list length")
		end
	end
	
	
	for i=1,#objectList do
		local object = objectList[i]
		
		if not parametersList then
			local ret = object[functionName](object)
			table.insert(resultList, ret)
		else
			local parameters = nil
			if #parametersList ~= #objectList then
				-- using the same parameters
				parameters = parametersList[1]
			else
				parameters = parametersList[i]
			end
			-- call function on object
			-- this equal to object:function(parameters)
			local ret = object[functionName](object, unpack(parameters))
			table.insert(resultList, ret)
		end
	end
	
	return resultList
end


