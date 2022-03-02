## Array Optimization using the Singular Values of the telescope operator

Author: Tim Molteno

See the jupyter notebook for details.

### INSTALL:

Issue the command

    make install

### RUN

Issue the command

    make

or have a look at the runtime options by running the array_opt.py command as follows:

    python3 array_opt.py --help

Sample output below

    usage: array_opt.py [-h] [--output OUTPUT] [--iter ITER] [--nant NANT] [--narm NARM] [--arcmin ARCMIN] [--radius RADIUS] [--radius-min RADIUS_MIN]
                        [--spacing SPACING] [--fov FOV] [--learning-rate LEARNING_RATE]

    DiSkO Array: Optimize an array layout using the singular values of the array operator

    optional arguments:
    -h, --help            show this help message and exit
    --output OUTPUT       Root of output file names. (default: optimized_array)
    --iter ITER           Number of iterations. (default: 100)
    --nant NANT           Number of antennas per arm. (default: 8)
    --narm NARM           Number of arms. (default: 3)
    --arcmin ARCMIN       Resolution of the sky in arc minutes. (default: 120)
    --radius RADIUS       Length of each arm in meters. (default: 2.0)
    --radius-min RADIUS_MIN
                            Minimum antenna position along each arm in meters. (default: 0.1)
    --spacing SPACING     Minimum antenna spacing. (default: 0.15)
    --fov FOV             Field of view in degrees (default: 180.0)
    --learning-rate LEARNING_RATE
                            Optimizer learning rate. (default: 0.02)



### TODO:

* Add the ability to start from a JSON file.

The following array reached 16.1:

        Arm 0: [0.224 0.344 0.787 1.292 1.600 1.928 2.864 3.121]
        Arm 120: [0.229 0.897 1.324 2.250 2.585 2.712 3.544 4.174]
        Arm 240: [0.094 0.106 1.001 1.158 1.273 1.575 2.148 3.008]
