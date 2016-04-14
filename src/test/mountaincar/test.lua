package.path = package.path..";../../?/init.lua"
package.path = package.path..";../../?.lua"


require 'control'


function finish()
	return false
end


function main()
	local mountainCar = init(false)
	
	while not finish() do
		step()
	end

end


main()