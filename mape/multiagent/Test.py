def update():
	v = 1
	left = False
	while True:
		if v> 10:
			left = True
		elif v<-10:
			left = False
		if left:
			v -= 1
			# yield v
		else:
			v += 1 
		yield v
while True:
	print("---->", update())