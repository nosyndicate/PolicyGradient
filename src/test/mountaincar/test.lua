package.path = package.path..";../../?/init.lua"
package.path = package.path..";../../?.lua"



require 'control'




function main()
	local mountainCar = init(false)

	local finished = false
	
	while not finished do
		finished = step(3000)
	end

end


main()