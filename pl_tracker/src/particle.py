from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn, uniform, seed
import scipy.stats

class ParticleFilter:
    ##### Initialization #####
    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi
        return particles


    ##### predict/update loop ##### 
    def predict(self, particles, u, std, dt=1.):
        N = len(particles) 
        p_dim = particles.shape[1] - 1
        # update heading
        particles[:, 2] += norm(u + (randn(N,p_dim) * (np.sqrt(std))))
        particles[:, 2] %= 2 * np.pi
        # particles[:, 2] = (np.arctan(dirs[:,1]/dirs[:,0]) + ((randn(N,p_dim)) * (np.sqrt(std)/10)))
        
        dirs = (u-particles[:,0:2]) + ((randn(N,p_dim)) * (np.sqrt(std)))
        dirs /= np.sqrt( np.fabs(dirs) )

        # move in the (noisy) commanded direction
        # dist = (dirs * dt) + (randn(N,p_dim) * (np.sqrt(std)/10))
        
        # particles[:, 0] += np.cos(particles[:, 2]) * dist[:,0]
        # particles[:, 1] += np.sin(particles[:, 2]) * dist[:,1]
        particles[:, 0] += dirs[:,0] #* dist[:,0]
        particles[:, 1] += dirs[:,1] #* dist[:,1]

        


    def update(self, particles, weights, z, R, landmarks):
        weights.fill(1.)
        for i, landmark in enumerate(landmarks):
            # dist = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            dist = particles[:, 0:2] - landmark
                        
            # print("zs: ", z, z.shape)
            # print("z: ", z[i], z[i].shape)
            # print("R: ", R, len(R))
            # print("landmarks: ", landmarks, len(landmarks))
            # print("Dist: ", dist, len(dist))
            # # weights *= scipy.stats.norm(dist, np.linalg.norm(R)).pdf(np.linalg.norm(z[i]))
            # print("Distribution: ", scipy.stats.norm(dist, R))
            probs = scipy.stats.norm(dist, R).pdf(np.linalg.norm(z[i]))
            # print("probz: ", probs)
            weights *= np.linalg.norm(probs, axis=1)

        # print("Weights: ", weights, len(weights))
        weights /= (sum(weights) + 1.e-300) # normalize(and avoid division by 0)


    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""

        pos = particles[:, 0:2]
        mean = np.average(pos, weights=(weights + 1.e-300), axis=0)
        var  = np.average((pos - mean)**2, weights=(weights + 1.e-300), axis=0)
        return mean, var


    ##### resample #####
    def resample(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights /= np.sum(weights)   


    def neff(self, weights):
        # print( "Weights:", weights)
        return 1. / (np.sum(np.square(weights)) + 1.e-30)



    ##### particle filter #####
    def run_pf1(N, iters=18, sensor_std_err=.1, 
                do_plot=True, plot_particles=False,
                xlim=(0, 20), ylim=(0, 20),
                initial_x=None):

        ## Landmarks
        landmarks = np.array([[5,0], [17, 5], [12,10], [5,5]])

        N_len = len(landmarks)

        plt.figure()

        # initialize particles and weights
        if initial_x is not None:
            particles = create_gaussian_particles(
                mean=initial_x, std=(5, 5, np.pi/4), N=N)
        else:
            particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
        weights = np.zeros(N)

        if plot_particles:
            alpha = .20
            if N > 5000:
                alpha *= np.sqrt(5000)/np.sqrt(N)           
            plt.scatter(particles[:, 0], particles[:, 1], 
                        alpha=alpha, color='g')

        ### Main loop ###
        xs = []
        robot_pos = np.array([0., 0.])
        for x in range(iters):
            robot_pos += (1, 1)
            # robot_pos += (x+1, 1)
            # robot_pos += (np.sin(x*np.pi/50+1), np.cos(x*np.pi/50))

            

            # distance from robot to each landmark
            zs = (norm(landmarks - robot_pos, axis=1) + 
                  (randn(N_len) * sensor_std_err))

            # move diagonally forward to (x+1, x+1)
            predict(particles, u=(0.0, 1.414), std=(.2, .05))

            # incorporate measurements
            update(particles, weights, z=zs, R=sensor_std_err, 
                   landmarks=landmarks)

            # resample if too few effective particles
            if neff(weights) < N/2:
                indexes = systematic_resample(weights)
                resample(particles, weights, indexes)

            mu, var = estimate(particles, weights)
            xs.append(mu)


            # Plot
            if plot_particles:
                plt.scatter(particles[:, 0], particles[:, 1], 
                            color='b', marker=',', s=1)
            p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                             color='k', s=180, lw=3)
            p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

        xs = np.array(xs)
        #plt.plot(xs[:, 0], xs[:, 1])
        plt.legend([p1, p2], ['Ground Truth', 'Particle Filter Estimate'], loc=4, numpoints=1)
        # plt.xlim(*xlim)
        # plt.ylim(*ylim)
        print('final position error, variance:\n\t', mu, var)
        plt.show()


# seed(42) 
# run_pf1(N=100, iters=20, plot_particles=True)

