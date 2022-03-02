import disko
import logging
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#    Use the SVD of the telescope operator to evaluate the performance of an all-sky array
#    layout
#



def polar_to_rectangular(r, theta):
    '''
        Rectangular coordinates are with the y axis aligned with north-south
        and the x axis pointing east
    '''
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    z = np.zeros_like(x)
    
    return x,y,z


def init(narms, nants, radius_lower, spacing, radius_limit):
    '''
        Define the globals needed by our minimization functoin
    '''
    global l, m, n_minus_1, p2j, theta, pixel_areas, radius, radius_min, min_spacing,  num_arms, arm_degrees, arm_angles, ants_per_arm,  N_smallest
    
    l = tf.constant(ant.fov.l)
    m = tf.constant(ant.fov.m)
    n_minus_1 = tf.constant(ant.fov.n_minus_1)
    
    freq = 1.57542e9
    C = 2.99793e8
    

    num_ant = nants
    num_arms = narms
    
    N_smallest = (num_ant*(num_ant - 1) // 2) - 1

    arm_indices = np.array_split(range(num_ant), num_arms)
    
    ants_per_arm = np.array([s.shape[0] for s in arm_indices])
    
    radius = tf.constant(radius_limit, dtype=tf.float64)
    min_spacing = tf.constant(spacing, dtype=tf.float64)
    radius_min = tf.constant(radius_lower, dtype=tf.float64)
    
    p2j = tf.constant(2.0*np.pi*1.0j * freq / C)

    
    arm_degrees = np.linspace(0,  360,  num_arms, endpoint=False)
    #arm_degrees = arm_degrees + np.random.uniform(-20,  20,  num_arms)
    arm_angles = np.radians(arm_degrees)
   
    print(f"Arm Angles {arm_angles}")
    print(f"Arm Nant {ants_per_arm}")
    arm_thetas = [[a]*n for a, n in zip(arm_angles, ants_per_arm)]
    arm_thetas = [x for sublist in arm_thetas for x in sublist]
    print(f"{arm_thetas}")
    theta = tf.constant(np.array(arm_thetas).flatten())

    n_s = l.shape[0]
    pixel_areas = tf.constant(1.0 / np.sqrt(n_s), dtype=tf.complex128)
    
def constrain(x, lower, upper):
    sharpness = 5
    clip_lower = tf.math.softplus((x-lower)*sharpness)/sharpness + lower
    return upper - tf.math.softplus((-clip_lower + upper)*sharpness)/sharpness

def penalize(duv, limit=0.2):
    sharpness = 40
    clip_lower = tf.math.softplus((limit - duv)*sharpness)/sharpness
    ret = 3* clip_lower/limit
    ret = ret*ret / duv
    #if (duv < limit):
        #print(f"penalize({duv}, {limit}) -> {ret}")
    return ret

def global_f(x):
    '''
        A function suitable for optimizing using a global minimizer. This will
        return the condition number of the telescope operator
    '''
    
    global l, m, n_minus_1, p2j, theta, pixel_areas, radius, radius_min, min_spacing

    x_constrained = constrain(x, lower=radius_min, upper=radius)

    _x = x_constrained * tf.sin(theta)
    _y = x_constrained * tf.cos(theta)
    _z = tf.zeros_like(x)
    
    num_ant = x.shape[0]
    
    rows = []
    penalty = 0
    for i in range(num_ant):
        for j in range(i+1, num_ant):
            u = _x[i] - _x[j]
            v = _y[i] - _y[j]
            w = _z[i] - _z[j]
            
            duv =tf.sqrt (u**2 + v**2 + w**2)
            penalty += penalize(duv, min_spacing)
            
            exponent = -p2j*tf.cast(u*l + v*m + w*n_minus_1, tf.complex128)
            h = tf.exp(exponent) * pixel_areas
            rows.append(h)
    
    gamma = tf.stack(rows, axis=0)

    s = tf.linalg.svd(gamma, full_matrices=False, compute_uv=False)
    score = (s[0] / s[N_smallest])
    print(f"C/N={score}  penalty={penalty} score={score+penalty}")
    return penalty, score

#Function without input
def fu_minimize():
    tf.debugging.check_numerics(x_opt, message="x is buggered")
    penalty, score = global_f(x_opt)
    return penalty + score


class YAntennaArray:
    '''
        Antenna array consists of three arms called arm0, arm120 and arm240.
        Each arm has N antennas. In the case of the tart, it will be N=8
        
    '''
    def __init__(self, N, radius, res_arcmin, fov_degrees):
        self.N = N
        self.radius = radius
        self.frequency = 1.57542e9
        self.fov = disko.HealpixSubSphere.from_resolution(resolution=res_arcmin, 
                                      theta = np.radians(0.0), phi=0.0, 
                                      radius=np.radians(fov_degrees/2))
        self.fig = plt.figure(figsize=(12,6))
        self.ax1 = self.fig.add_subplot(1,2,1, adjustable='box', aspect=1)
        self.ax2 = self.fig.add_subplot(1,2,2, adjustable='box', aspect=1)

        #self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        #self.fig.suptitle('Optimizer Outpu')
        
    def score(self, arms):
        # Create telescope operator
        
        dsko = self.get_disko(arms)
        
        gamma = dsko.make_gamma(self.fov)
        print("gamma shape {}".format(gamma.shape))
        s = tf.linalg.svd(gamma, full_matrices=False, compute_uv=False)

        score =  s[0] / s[N_smallest]
        return score

    def get_ant_pos(self, arms):
        
        ant_pos = np.hstack([polar_to_rectangular(r=r, theta=a) for r, a in zip(arms, arm_angles)]).T            
        return ant_pos

    def get_disko(self, arms):
        frequencies = [self.frequency]
        
        ant_pos = self.get_ant_pos(arms)
                
        array_disko = disko.DiSkO.from_ant_pos(ant_pos, frequencies[0])
        return array_disko
    
    def sort_arms(self, arms):
        ret = [np.sort(a) for a in arms]
        return ret
    
    def plot_uv(self, arms, score, penalty, output):
        
        lim = radius*1.1
        #self.fig.clf()
        self.ax1.clear()
        self.ax1.set_aspect('equal', adjustable='box')
        dsko = self.get_disko(arms)
        self.ax1.set_title("U-V coverage C={:.3g}".format(score))
        self.ax1.set_ylim(-2*radius, 2*radius)
        self.ax1.set_xlim(-2*radius, 2*radius)
        self.ax1.plot(dsko.u_arr, dsko.v_arr, '.')
        self.ax1.plot(-dsko.u_arr, -dsko.v_arr, '.')
        self.ax1.set_xlabel('u (m)')
        self.ax1.set_ylabel('v (m)')
        self.ax1.grid(True)
        
        ant_pos = self.get_ant_pos(arms)

        self.ax2.clear()
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax2.set_title(f"Antenna Locations (penalty {penalty :4.2f})")
        #self.ax2.plot(ant_pos[:,0], ant_pos[:,1], 'x')
        x_array = ant_pos[:,0]
        y_array = ant_pos[:,1]
        for x,y in zip(x_array, y_array):
            circle = plt.Circle((x, y), radius=0.075)
            self.ax2.add_patch(circle)
        self.ax2.set_ylim(-lim, lim)
        self.ax2.set_xlim(-lim, lim)
        self.ax2.set_xlabel('x (m)')
        self.ax2.set_ylabel('y (m)')
        self.ax2.grid(True)

        self.fig.tight_layout()
        self.fig.savefig(f"{output}.pdf")
        self.fig.savefig(f"{output}.png")

        plt.pause(0.1)
        
        ret = {}
        ret['C/N'] = score
        ret['penalty'] = penalty
        ret['arm_degrees'] = arm_degrees.tolist()
        
        sorted_arms = self.sort_arms(arms)
        arm_array = [a.tolist() for a in sorted_arms]
        ret["arms"] = arm_array

        fname = f"{output}.json"
        with open(fname, 'w') as outfile:
            json.dump(ret, outfile, sort_keys=True, indent=4)
            
    
    def print(self, arms):
        sorted_arms = self.sort_arms(arms)
        for a, d  in zip(sorted_arms, arm_degrees):
            s = np.array2string(a, formatter={'float_kind':lambda x: "%.3f" % x})
            print(f"    Arm {d}: {s}")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='DiSkO Array: Optimize an array layout using the singular values of the array operator', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output', default='optimized_array', help="Root of output file names.")

    parser.add_argument('--iter', type=int, default=100, help="Number of iterations.")
    parser.add_argument('--nant', type=int, default=8, help="Number of antennas per arm.")
    parser.add_argument('--narm', type=int, default=3, help="Number of arms.")

    parser.add_argument('--arcmin', type=float, default=120, help="Resolution of the sky in arc minutes.")
    parser.add_argument('--radius', type=float, default=2.0, help="Length of each arm in meters.")
    parser.add_argument('--radius-min', type=float, default=0.1, help="Minimum antenna position along each arm in meters.")
    parser.add_argument('--spacing', type=float, default=0.15, help="Minimum antenna spacing.")

    parser.add_argument('--fov', type=float, default=180.0, help="Field of view in degrees")

    parser.add_argument('--learning-rate', type=float, default=0.02, help="Optimizer learning rate.")

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


    
    radius = ARGS.radius
    radius_min = ARGS.radius_min
    N = ARGS.nant
    
    ant = YAntennaArray(N=N, radius=radius, 
                        res_arcmin=ARGS.arcmin,
                        fov_degrees=ARGS.fov)
    best_score = 1e49
    
    # Set up global variables for the tf function
    init(ARGS.narm, N, radius_min, ARGS.spacing, radius)
    
    
    if True:
        x_opt = tf.Variable(tf.random_uniform_initializer(minval=radius_min,  maxval=radius)(shape=(N,),
                                                                              dtype=tf.float64))
        #x_opt =  tf.Variable(tf.linspace(start=radius_min, stop = radius, num=24))
                                                          
        opt = tf.keras.optimizers.RMSprop(learning_rate=ARGS.learning_rate)
        for i in range(ARGS.iter):
            opt.minimize(fu_minimize, var_list=[x_opt])
            penalty, score = global_f(x_opt)
            y = penalty.numpy() + score.numpy()
            #print (opt.get_gradients(y, [x_opt]))
            if (y < best_score):
                x_constrained = constrain(x_opt, radius_min, radius).numpy()
                arms = np.array_split(x_constrained,  num_arms)
                ant.print(arms)
                ant.plot_uv(arms, score.numpy(), penalty.numpy(), ARGS.output)
                best_score = y
                

    else:
        for i in range(ARGS.iter):
            arms = [np.random.uniform(0, radius, n) for a, n in zip(arm_angles, ants_per_arm) ]
            
            score = ant.score(arms)
            x = np.array(arms).flatten()
            score2 = global_f(x)
            print(score, score2)
            
            if (score < best_score):
                print("Iteration {} New best score {}".format(i, score))
                ant.print(arms)
                best_score = score
                ant.plot_uv(arms, score, penalty)
        
        print("Best score: {}".format(best_score))
        ant.print(arms)
