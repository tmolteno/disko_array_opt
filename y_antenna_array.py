import disko
import json

import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf


def polar_to_rectangular(r, theta):
    '''
        Rectangular coordinates are with the y axis aligned with north-south
        and the x axis pointing east
    '''
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    z = np.zeros_like(x)
    
    return x,y,z


class YAntennaArray:
    '''
        Antenna array consists of three arms called arm0, arm120 and arm240.
        Each arm has N antennas. In the case of the tart, it will be N=8
        
    '''
    def __init__(self, N, narms, radius, res_arcmin, fov_degrees, radius_lower, spacing):
        self.N = N
        self.radius = radius
        self.frequency = 1.57542e9
        self.res_arcmin = res_arcmin
        self.fov_degrees = fov_degrees
        
        self.fov = disko.HealpixSubSphere.from_resolution(res_arcmin=res_arcmin, 
                                      theta = np.radians(0.0), phi=0.0, 
                                      radius_rad=np.radians(fov_degrees/2))
        self.fig = plt.figure(figsize=(18,6))
        self.ax1 = self.fig.add_subplot(1,3,1, adjustable='box', aspect=1)
        self.ax2 = self.fig.add_subplot(1,3,2, adjustable='box', aspect=1)
        self.ax3 = self.fig.add_subplot(1,3,3, adjustable='box', aspect=1)

        #self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        #self.fig.suptitle('Optimizer Outpu')

        self.l = tf.constant(self.fov.l)
        self.m = tf.constant(self.fov.m)
        self.n_minus_1 = tf.constant(self.fov.n_minus_1)
        
        freq = 1.57542e9
        C = 2.99793e8
        

        num_ant = N
        self.num_arms = narms
        
        #self.N_smallest = (num_ant*(num_ant - 1) // 2) - 1
        self.N_smallest = (num_ant*(num_ant - 1)) - 1

        arm_indices = np.array_split(range(num_ant), self.num_arms)
        
        self.ants_per_arm = np.array([s.shape[0] for s in arm_indices])
        
        self.radius_limit = tf.constant(self.radius, dtype=tf.float64)
        self.min_spacing = tf.constant(spacing, dtype=tf.float64)
        self.radius_lower = tf.constant(radius_lower, dtype=tf.float64)
        
        wavelength = C / freq
        omega = 2.0*np.pi/wavelength
        self.p2j = 1.0j*omega

        
        self.arm_degrees = np.linspace(0,  360,  self.num_arms, endpoint=False)
        #arm_degrees = arm_degrees + np.random.uniform(-20,  20,  self.num_arms)
        self.arm_angles = np.radians(self.arm_degrees)
    
        print(f"Arm Angles {self.arm_angles}")
        print(f"Arm Nant {self.ants_per_arm}")
        arm_thetas = [[a]*n for a, n in zip(self.arm_angles, self.ants_per_arm)]
        arm_thetas = [x for sublist in arm_thetas for x in sublist]
        print(f"Arm Angles {arm_thetas}")
        self.theta = tf.constant(np.array(arm_thetas).flatten())

        n_s = self.l.shape[0]
        self.pixel_areas = tf.constant(1.0 / np.sqrt(n_s), dtype=tf.complex128)
        
    def condition_number(self, arms):
        # Create telescope operator
        
        dsko = self.get_disko(arms)
        
        gamma = dsko.make_gamma(self.fov, makecomplex=True)
        print("gamma shape {}".format(gamma.shape))
        s = tf.linalg.svd(gamma, full_matrices=False, compute_uv=False)
        c_n =  s[0] / s[self.N_smallest]
        
        eps = 1e-14
        sigma = s[0:self.N_smallest]
        snorm = sigma / s[0] # tf.math.reduce_sum(sigma) # Normalized singular values
        eps = tf.constant(1e-14, dtype=tf.float64)
        entropy = tf.math.reduce_sum(-tf.math.log(snorm + eps))  # This is the log of the determinant of the moore penrose inverse
        condition_number = 10*tf.math.log(s[0] / (s[self.N_smallest] + eps))

        return entropy, condition_number, s


    def get_ant_pos(self, arms):
        ant_pos = np.hstack([polar_to_rectangular(r=r, theta=a) for r, a in zip(arms, self.arm_angles)]).T            
        return ant_pos

    def get_disko(self, arms):
        frequencies = [self.frequency]
        
        ant_pos = self.get_ant_pos(arms)
                
        array_disko = disko.DiSkO.from_ant_pos(ant_pos, frequencies[0])
        return array_disko
    
    def sort_arms(self, arms):
        ret = [np.sort(a) for a in arms]
        return ret
    
    def psf(self, arms):
        dsko = self.get_disko(arms)
        u = dsko.u_arr
        v = dsko.v_arr
        uvmax = max(np.max(u), np.max(v))
        
        grid_size = 128
        g2 = grid_size//2
        uv_plane = np.zeros((grid_size, grid_size))
        mask = np.ones((grid_size, grid_size))
        mask[g2-10:g2+10,g2-10:g2+10] = 0
        
        try:
            for u,v in zip(dsko.u_arr, dsko.v_arr):
                i = g2 + int(u*g2/uvmax - 0.5)
                j = g2 + int(v*g2/uvmax - 0.5)
                uv_plane[i,j] += 1
                #i = g2 - int(u*g2/uvmax - 0.5)
                #j = g2 - int(v*g2/uvmax - 0.5)
                #uv_plane[i,j] += 1
        
            psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_plane))))
            peak = np.max(psf)
            psf = psf / peak
            peak2 = np.max(psf*mask)
            self.ax3.clear()
            self.ax3.set_title(f"Point Spread Function: Peak radio: {1/peak2:4.2f}")
            img = self.ax3.imshow(psf*mask)
            fig.colorbar(img, ax=ax3, location='right', orientation='vertical')
        except:
            pass
        
    def plot_uv(self, arms, penalty, entropy, condition_number, output):
        
        lim = self.radius*1.1
        #self.fig.clf()
        self.ax1.clear()
        self.ax1.set_aspect('equal', adjustable='box')
        dsko = self.get_disko(arms)
        self.ax1.set_title(f"U-V coverage C={condition_number:4.2f} E={entropy:5.1f}")
        self.ax1.set_ylim(-2*self.radius, 2*self.radius)
        self.ax1.set_xlim(-2*self.radius, 2*self.radius)
        self.ax1.plot(dsko.u_arr, dsko.v_arr, '.')
        #self.ax1.plot(-dsko.u_arr, -dsko.v_arr, '.')
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
            circle = plt.Circle((x, y), radius=self.min_spacing/2)
            self.ax2.add_patch(circle)
        self.ax2.set_ylim(-lim, lim)
        self.ax2.set_xlim(-lim, lim)
        self.ax2.set_xlabel('x (m)')
        self.ax2.set_ylabel('y (m)')
        self.ax2.grid(True)

        self.psf(arms)
        
        self.fig.tight_layout()
        self.fig.savefig(f"{output}.pdf")
        self.fig.savefig(f"{output}.png")

        plt.pause(0.1)
        ret = self.to_dict(arms)
        ret['C/N'] = condition_number
        ret['entropy'] = entropy
        ret['penalty'] = penalty

        fname = f"{output}.json"
        with open(fname, 'w') as outfile:
            json.dump(ret, outfile, sort_keys=True, indent=4)
            
    def to_dict(self, arms):
        ret = {}
        ret['num_arms'] = self.num_arms
        ret['arm_degrees'] = self.arm_degrees.tolist()
        ret['radius'] = self.radius
        ret['radius_lower'] = self.radius_lower.numpy()
        ret['res_arcmin'] = self.res_arcmin
        ret['fov_degrees'] = self.fov_degrees
        ret['spacing'] = self.min_spacing.numpy()
        
        sorted_arms = arms # self.sort_arms(arms)
        arm_array = [a.tolist() for a in sorted_arms]
        ret["arms"] = arm_array
        return ret


    @classmethod
    def from_json(self, json_string):
        arms_list = json_string['arms']
        arms = [np.array(a) for a in arms_list]
        N = np.concatenate(arms).shape[0]
        narms = json_string['num_arms']
        radius = json_string['radius']
        res_arcmin = json_string['res_arcmin']
        fov_degrees = json_string['fov_degrees']
        radius_lower = json_string['radius_lower']
        spacing = json_string['spacing']
        ant = YAntennaArray(N, narms = narms,
                            radius=radius, 
                            res_arcmin=res_arcmin,
                            fov_degrees=fov_degrees,
                            radius_lower=radius_lower, 
                            spacing=spacing)
        
        ant.arms = arms
        return ant

    def print(self, arms):
        #print(f"Unsorted {arms}")
        sorted_arms = self.sort_arms(arms)
        for a, d  in zip(sorted_arms, self.arm_degrees):
            s = np.array2string(a, formatter={'float_kind':lambda x: "%.3f" % x})
            print(f"    Arm {d}: {s}")

