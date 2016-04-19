utils = {}


function utils.inTable(table, item)
    for key, value in pairs(table) do
        if value == item then
        	return key
        end
    end
    return false
end