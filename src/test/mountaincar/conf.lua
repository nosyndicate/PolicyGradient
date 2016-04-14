-- Configuration

function love.conf(t)
	t.title = "Mountain Car"

	t.window.width = 600
	t.window.height = 500
	
	-- attach a console for debugging
	t.console = false
	
	-- not resizable
	t.resizable = false
	
end