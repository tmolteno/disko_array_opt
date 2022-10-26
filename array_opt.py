import disko
import logging
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from y_antenna_array import YAntennaArray

#    Use the SVD of the telescope operator to evaluate the performance of an all-sky array
#    layout
#
    
    
def constrain(x, lower, upper):
    sharpness = 10
    clip_lower = tf.math.softplus((x-lower)*sharpness)/sharpness + lower
    return upper - tf.math.softplus((-clip_lower + upper)*sharpness)/sharpness

def penalize(duv, limit=0.25):
    sharpness = 10
    clip_lower = tf.math.softplus((limit - duv)*sharpness)/sharpness
    return clip_lower/limit

def penalize_below(x, limit, width):
    # Scale so that duv is expressed in multiples of width
    xs = x / width
    ls = limit / width
    return 10*limit*penalize(xs, ls)/width

def penalize_above(x, limit=0.2):
    sharpness = 40
    clip_lower = tf.math.softplus((x-limit)*sharpness)/sharpness   # Always positive
    ret = 100 * clip_lower
    return ret


def global_f(x):
    '''
        A function suitable for optimizing using a global minimizer. This will
        return the condition number of the telescope operator
    '''
    
    global ant 

    x_constrained = constrain(x, lower=ant.radius_lower, upper=ant.radius_limit)
    r_positive = x_constrained
    # r_positive = tf.concat([[0], tf.math.exp(x)],0)

    _x = r_positive * tf.sin(ant.theta)
    _y = r_positive * tf.cos(ant.theta)
    _z = tf.zeros_like(r_positive)
    
    num_ant = r_positive.shape[0]
    
    rows = []
    penalty = 0 # tf.linalg.norm(x - x_constrained)**2
    #penalty += tf.reduce_sum(penalize_above(r_positive, ant.radius_limit))**2  # ensure x[i] < radius_limit
    #penalty += tf.reduce_sum(penalize_below(r_positive, ant.radius_lower))**2  # x[i] > radius_lower
    
    for i in range(num_ant):
        
        #penalty += penalize_above(r_positive[i], ant.radius_limit)  # ensure x[i] < radius_limit
        #penalty += penalize_below(r_positive[i], ant.radius_lower)  # x[i] > radius_lower
        #for j in range(i+1, num_ant):
        for j in range(num_ant):
            if (i != j):
                u = _x[i] - _x[j]
                v = _y[i] - _y[j]
                w = _z[i] - _z[j]
                
                duv = tf.sqrt (u**2 + v**2 + w**2)
                penalty += penalize_below(duv, ant.min_spacing, width=0.01)
                
                exponent = ant.p2j*tf.cast(u*ant.l + v*ant.m + w*ant.n_minus_1, tf.complex128)
                h = tf.math.exp(exponent) * ant.pixel_areas
                rows.append(h)
                
                #h2 = tf.math.conj(h)
                #rows.append(h2)
    
    gamma = tf.stack(rows, axis=0)

    s = tf.linalg.svd(gamma, full_matrices=False, compute_uv=False)
    
    '''
        https://math.stackexchange.com/questions/542035/what-does-svd-entropy-capture
        
        We use the entropy of the Moore-Penrose Inverse as the measure. (https://arxiv.org/abs/1110.6882)
        
        Condition number: (1/smallest) / (1/largest) = 1 / (smallest/largest) = largest/smallest
    '''
    sigma = s[0:ant.N_smallest]
    snorm = sigma / s[0] # tf.math.reduce_sum(sigma) # Normalized singular values
    eps = tf.constant(1e-14, dtype=tf.float64)
    entropy = tf.math.reduce_sum(-tf.math.log(snorm + eps))  # This is the log of the determinant of the moore penrose inverse
    condition_number = 10*tf.math.log(s[0] / (s[ant.N_smallest] + eps))

    return penalty, entropy, condition_number

def criterion(p, e, c_n):
    global ARGS
    if ARGS.entropy:
        return (p + e)
    else:
        return (p + c_n)

#Function without input
def fu_minimize():
    global penalty, entropy, condition_number
    tf.debugging.check_numerics(x_opt, message="x is buggered")
    penalty, entropy, condition_number = global_f(x_opt)
    return criterion(penalty, entropy, condition_number)



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='DiSkO Array: Optimize an array layout using the singular values of the array operator', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output', default='optimized_array', help="Root of output file names.")

    parser.add_argument('--input', default=None, help="Input initial positions.")

    parser.add_argument('--iter', type=int, default=100, help="Number of iterations.")
    parser.add_argument('--nant', type=int, default=8, help="Number of antennas per arm.")
    parser.add_argument('--narm', type=int, default=3, help="Number of arms.")

    parser.add_argument('--entropy', action='store_true', help="Optimize on Entropy rather than C/N.")

    parser.add_argument('--arcmin', type=float, default=120, help="Resolution of the sky in arc minutes.")
    parser.add_argument('--radius', type=float, default=2.0, help="Length of each arm in meters.")
    parser.add_argument('--radius-min', type=float, default=0.1, help="Minimum antenna position along each arm in meters.")
    parser.add_argument('--spacing', type=float, default=0.15, help="Minimum antenna spacing.")

    parser.add_argument('--fov', type=float, default=180.0, help="Field of view in degrees")

    parser.add_argument('--learning-rate', type=float, default=0.02, help="Optimizer learning rate.")
    parser.add_argument('--optimizer', default='RMSprop', help="Optimization algorithm.")

    ARGS = parser.parse_args()


    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='array_opt.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    
    # add the handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(console)


    
    # Set up global variables for the tf function
    
    #glob = OptGlobals(ARGS.narm, N, radius_min, ARGS.spacing, radius)
    ant = YAntennaArray(N=ARGS.nant, 
                        narms = ARGS.narm,
                        radius=ARGS.radius, 
                        res_arcmin=ARGS.arcmin,
                        fov_degrees=ARGS.fov,
                        radius_lower=ARGS.radius_min, 
                        spacing=ARGS.spacing)
    
    best_score = 1e49
    
    
    hist_json = {}
    hist_json['y'] = []
    hist_json['penalty'] = []
    hist_json['entropy'] = []
    hist_json['c_n'] = []
    
    penalty = 1e50
    entropy = 1e50
    condition_number = 1e50
    
    l_rate = ARGS.learning_rate
    momentum = ARGS.learning_rate
    dither = ant.min_spacing/20

    if ARGS.input is not None:
        with open(ARGS.input, 'r') as infile:
            x0_json = json.load(infile)

        x0_arms = x0_json['arms']
        
        x0 = np.concatenate([np.array(a) for a in x0_arms])
        print(f"x0 = {x0}")
        x_opt = tf.Variable(x0)
    else:
        x_opt = tf.Variable(tf.random_uniform_initializer(minval=(tf.constant(ant.radius_lower, dtype=tf.float64)),  
                                                          maxval=(tf.constant(ant.radius, dtype=tf.float64)))(shape=(ant.N,),  dtype=tf.float64))
    
    if (ARGS.optimizer=='SGD'):
        opt = tf.keras.optimizers.SGD(learning_rate=l_rate, momentum=0)
    elif (ARGS.optimizer=='Adamax'):
        opt = tf.keras.optimizers.Adamax(learning_rate=l_rate)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=l_rate, momentum=momentum)
    for i in range(ARGS.iter):
        opt.minimize(fu_minimize, var_list=[x_opt])
        r_positive = x_opt
        
        y = criterion(penalty.numpy(), entropy.numpy(), condition_number.numpy())
        hist_json['y'].append(y)
        hist_json['c_n'].append(condition_number.numpy())
        hist_json['entropy'].append(entropy.numpy())
        hist_json['penalty'].append(penalty.numpy())
        
        print(f"score={y:4.2f}  penalty={penalty:4.2f} C/N={condition_number:4.2f}, Entropy={entropy:4.2f}")
        if (y < best_score):
            x_constrained = constrain(x_opt, ant.radius_lower, ant.radius_limit).numpy()
            #print(f"constrained: {x_constrained}")
            #print(f"unconstrain: {x_opt.numpy()}")
            arms = np.array_split(x_constrained,  ant.num_arms)
            ant.print(arms)
            ant.psf(arms)
            ant.plot_uv(arms, penalty.numpy(), entropy.numpy(), condition_number.numpy(), ARGS.output)
            best_score = y
        
        fname = f"{ARGS.output}_history.json"
        with open(fname, 'w') as outfile:
            json.dump(hist_json, outfile, sort_keys=True, indent=4)
