NANT=24
FOV=160
ARCMIN=180
SPACING=0.20
RADIUS_MIN=0.15

COMMON_OPTS=--nant=${NANT} --spacing=${SPACING} --fov=${FOV} --arcmin=${ARCMIN} --radius-min=${RADIUS_MIN}

#OPTS=${COMMON_OPTS} --optimizer=RMSprop --learning-rate=0.005 --iter=10000
OPTS=${COMMON_OPTS} --optimizer=SGD   --entropy  --learning-rate=0.000002 --iter=10000

analyze:
	python3 array_analyze.py --input three_arm_opt.json --iter 0

seven:
	python3 array_opt.py --output "seven_arm" --narm=7 ${OPTS} --radius=2.2
	
stellenbosch:
	python3 array_opt.py --output "stellenbosch"  --narm=5 ${OPTS} --radius=2.5
	
rhodes:
	python3 array_opt.py --output "rhodes" --narm=3 ${OPTS} --nant=23 --radius=2.6 --input='rhodes_opt.json'

three:
	python3 array_opt.py --output "three_arm" --narm=3 ${OPTS} --radius=2.6 --input='three_arm_opt.json'

cad:
	python3 make_scad.py --json 'three_arm_opt.json'

install:
	sudo aptitude install python3-numpy python3-scipy
# 	sudo pip3 install disko --upgrade
	pip3 install tensorflow --upgrade
