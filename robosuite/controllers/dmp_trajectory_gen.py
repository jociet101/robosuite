import numpy as np
import math


class DMPTrajectoryGenerator:
    def __init__(self,
                 tau,
                 alpha,
                 beta,   
                 num_dims,
                 num_basis,
                 num_sensors,
                 add_min_jerk=True,
                 mu=None,
                 gamma=None,
                 use_goal_formulation=False,
                 ) -> None:
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.num_dims = num_dims
        self.num_basis = num_basis
        self.num_sensors = num_sensors
        self.add_min_jerk = add_min_jerk
        self.use_goal_formulation = use_goal_formulation

        self.phi_ij = np.ones((self.num_dims, self.num_sensors))

        # updated at each run
        self.weights = None
        self.y0 = None
        self.dt = None
        self.T = None

        if mu is None:
            mu, gamma = self.get_default_gaussian_basis()
        self.mu, self.gamma = mu, np.array(gamma)
        self.expand_gaussian_basis_functions()

    def set_default_gaussian_basis(self):
        '''Set gaussian kernels with default parameters as basis functions.'''
        num_dims, num_basis = self.num_dims, self.num_basis
        num_sensors = self.num_sensors

        mu = np.array([np.exp(-i * (0.5 / (num_basis - 1))) for i in range(num_basis)])
        ## NOTE: This is not the std, but inverse_std, since below we multiply it with (x - mu)**2.
        gamma = [0.5 / (0.65 * (mu[i+1] - mu[i])**2) for i in range(num_basis - 1)]
        gamma += [gamma[-1]]

        return mu, np.array(gamma)
    
    def expand_gaussian_basis_functions(self):
        num_dims, num_basis = self.num_dims, self.num_basis
        num_sensors = self.num_sensors

        # Get mu and h for all parameters separately
        self.mu_all = np.zeros((num_dims, num_sensors, num_basis))
        self.h_all = np.zeros((num_dims, num_sensors, num_basis))
        for i in range(num_dims):
            for j in range(num_sensors):
                self.mu_all[i, j] = self.mu
                self.h_all[i, j] = self.gamma

    def run_with_weights(self, weights, y0, dt, traj_time, goal=None):
        '''Example to run the dmp.'''
        self.init_with_weights(weights, y0, dt, traj_time, goal=goal)
        x = self.xlog[-1]

        for i in range(self.T - 1):
            dmp_step_results = self.step(x, i, self.y[i], self.dy[i], phi_j=None)
            self.y.append(dmp_step_results['y'])
            self.dy.append(dmp_step_results['dy'])
            x = self.update_x(x, dt)
            self.update_x(x)
        
    def get_x(self):
        '''Get latest canonical variable value.'''
        return self.xlog[-1]

    def update_x(self, x, dt):
        '''Update the canonical system'''
        x = x + ((-self.tau * x) * dt)
        if (x < self.mu[-1] - 3.0*np.sqrt(1.0/(self.gamma[-1]))):
            x = 1e-7
        return x
    
    def set_x(self, x):
        '''Update canonical variable value.'''
        return self.xlog.append(x)
        
    def init_with_weights(self, weights, y0, dt, traj_time, goal=None):
        self.weights = weights
        self.y0 = y0
        self.dt = dt
        self.T = traj_time

        if self.use_goal_formulation:
            assert goal is not None, "Invalid goal for DMP with goal formulation"
            self.goal = np.array(goal)

        self.y = [y0]
        self.dy = [np.zeros_like(y0)]
        self.xlog = [1]

        self.min_jerk_log = []
    
    def step(self, x, step_idx, last_y, last_dy, phi_ij=None):
        '''Take a forward integration step with dmp params.
        
        x: State of canonical variable.
        phi_ij: Amplitude terms.
        '''
        # First calculate forcing function
        # For now we assume that we only have gaussian basis functions
        psi_ijk = np.exp(-self.h_all * (x-self.mu_all)**2)
        psi_ij_sum = np.sum(psi_ijk, axis=2, keepdims=True)
        f = (psi_ijk * self.weights[:, :, 1:] * x).sum(
                axis=2, keepdims=True) / (psi_ij_sum + 1e-10)
            
        # Now calculate min-jerk's force term
        f_min_jerk = min(-np.log(x)*2, 1)
        # x^3 * (6x^2 = 15*x + 10)  See 5.2 here: https://arxiv.org/pdf/2102.07459.pdf
        f_min_jerk = (f_min_jerk**3)*(6*(f_min_jerk**2) - 15*f_min_jerk+ 10)
        psi_ij_jerk = self.weights[:, :, 0:1] * f_min_jerk

        # for debug
        self.min_jerk_log.append(f_min_jerk)

        # Calculate the final force values per basis function.
        # calculate f(x; w_j)l -- shape (N, M)
        all_f_ij = self.alpha * self.beta * (f + psi_ij_jerk).squeeze()

        if phi_ij is None:
            phi_ij = self.phi_ij
        assert phi_ij.shape == (self.num_dims, self.num_sensors), f"Invalid shape for amplitude params. {phi_ij.shape}"

        if len(phi_ij.shape) == 2:
            if len(all_f_ij.shape) == 1:
                all_f_ij = all_f_ij.reshape(-1, 1)
            assert all_f_ij.shape == phi_ij.shape
            all_f_i = np.sum(all_f_ij * phi_ij, axis=1)
        else:
            raise ValueError(f"Incorrect shape for phi_ij (amplitude params): {phi_ij.shape}")

        if self.use_goal_formulation:
            ddy = self.alpha*(self.beta*(self.goal - last_y) - last_dy/self.tau) + all_f_i
        else:
            ddy = self.alpha*(self.beta*(self.y[0] - last_y) - last_dy/self.tau) + all_f_i
        ddy = ddy * (self.tau ** 2)

        dy = last_dy + ddy * self.dt
        y  = last_y + dy * self.dt

        return dict(
            y=y,
            dy=dy,
            psi_ijk=psi_ijk,
            force_min_jerk=f_min_jerk,
        )