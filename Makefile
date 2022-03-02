NANT=24
FOV=170
ARCMIN=180


seven:
	python3 array_opt.py --output "seven_arm" --nant=${NANT} --narm=7 --iter=10000 --fov=${FOV} --arcmin=${ARCMIN} \
		--radius-min=0.2 --radius=2.2 --learning-rate=0.02
five:
	python3 array_opt.py --output "five_arm" --nant=${NANT} --narm=5 --iter=10000 --fov=${FOV} --arcmin=${ARCMIN} \
		--radius-min=0.2 --radius=2.2 --learning-rate=0.02

three:
	python3 array_opt.py --output "three_arm" --nant=${NANT} --narm=3 --iter=10000 --fov=${FOV} --arcmin=${ARCMIN} \
		--radius-min=0.15 --radius=2.4 --learning-rate=0.02

cad:
	python3 make_scad.py --json 'five_arm.json'

install:
	sudo aptitude install python3-numpy python3-scipy
	sudo pip3 install disko --upgrade
	sudo pip3 install tensorflow --upgrade
