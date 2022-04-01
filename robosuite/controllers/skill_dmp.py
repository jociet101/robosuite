import numpy as np
import copy

from .skills import BaseSkill
from robosuite.controllers.dmp_trajectory_gen import DMPTrajectoryGenerator

class DMPPositionSkill(BaseSkill):
    def __init__(self, 
                 skill_type,
                 max_action_calls: int = 40,
                 params_key='',
                 use_axes=False,
                 axes=None,
                 dt=0.1,
                 add_weight_exploration=False,
                 **kwargs):
        super().__init__(
            skill_type=skill_type,
            use_gripper_params=False,
            use_ori_params=False,
            max_action_calls=max_action_calls,
            **kwargs,
        )
        self.add_weight_exploration = add_weight_exploration
        self.max_action_calls = max_action_calls
        self.params_key = params_key
        self.use_axes = use_axes
        self.axes = axes
        if not self.use_axes:
            self.axes = ['x', 'y', 'z']
        self.dt = dt
        self.rl_param_info = {'continuous': {'size':15, 'low':[], 'high':[], 'scale':False}, 'discrete':{'size':[]}}

    def get_param_dim(self, base_param_dim):
        return self.get_raw_param_dim()
    
    def get_raw_param_dim(self):
        return self.rl_param_info['continuous']['size']

    def log_parameters(self):
        '''Parameters to log for debugging what actions are being taken.'''
        size = self.get_raw_param_dim()
        if self.add_weight_exploration:
            values = self._params[:size].tolist()
        else:
            values = [0] * size
        names = [f'p_{i:02d}' for i in range(size)]
        return dict(values=values, size=size, names=names)

    def histogram_bins_to_log_parameters(self):
        num_params = self.get_raw_param_dim()
        low, high = -10 * np.ones(num_params,), 10 * np.ones(num_params)
        return np.split(np.linspace(low, high, 21), num_params, axis=1)
    
    def base_reset(self, config_update=None):
        # This will add robot_controller and similar vars to the config.
        if config_update is not None:
            self._config.update(config_update)
    
    def reset(self, params, start_pos, goal_pos, config_update=None):
        self.base_reset(config_update=config_update)

        # self.task_space, self.null_space = get_simple_projection_spaces(self.axes)
        self.task_space = np.eye(3)

        # dmp_params = info[self.params_key]
        self._dmp_traj_gen = DMPTrajectoryGenerator(
            params['tau'],
            params['alpha'],
            params['beta'],
            params['num_dims'],
            params['num_basis'],
            params['num_sensors'],
            add_min_jerk=params['add_min_jerk'],
            mu=params['mean'],
            gamma=params['gamma'],
            use_goal_formulation=params['use_goal_formulation'],
        )
        self.weights = np.copy(params['weights'])

        # Add params to weights
        # self.weight_params = np.zeros_like(self.weights)
        self.weight_params = np.copy(self.weights)
        # last_param_idx = 0
        # if 'x' in self.axes:
        #     self.weight_params[0, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        # if 'y' in self.axes:
        #     self.weight_params[1, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        # if 'z' in self.axes:
        #     self.weight_params[2, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        
        self.goal_pos = goal_pos
        # self._goal_pos = self._get_reach_pos(info)
        # self.dt = 0.05 * 5
        self._dmp_traj_gen.init_with_weights(
            self.weights,
            # info['cur_ee_pos'],
            start_pos,
            self.dt,
            self.max_action_calls - 4,
            goal=self.goal_pos,
        )
        self.step = 0
    
    def get_pos_action(self, info):
        is_delta = False
        pos = np.array(self._dmp_traj_gen.y[self.step])

        # if task space projection makes something 0 (e.g. z) then its min bound may lie above.
        # if self.clip_xyz_bounds:
        #     pos = np.clip(pos, self.global_xyz_bounds[0, :], self.global_xyz_bounds[1, :])

        if self.use_axes:
            pos = self.task_space @ pos

        return pos, is_delta
    
    def get_ori_action(self, info):
        # ori = super().get_ori_ac({'params':[0,0,0,0,0,0,0]})
        ori = [0.0,0.0,0.0]
        # ori[:] = 0.0
        is_delta = True
        return ori, is_delta

    def get_gripper_action(self, info):
        return [0]
        # raise NotImplementedError

    def update_state(self, info):
        x = self._dmp_traj_gen.get_x()
        i = self.step

        # phi_ij would be any object based amplitude terms that can shape the trajectory
        # through the forcing function.
        dmp_step_results = self._dmp_traj_gen.step(
            x, i, self._dmp_traj_gen.y[i], self._dmp_traj_gen.dy[i], phi_ij=None)

        self._dmp_traj_gen.y.append(dmp_step_results['y'])
        self._dmp_traj_gen.dy.append(dmp_step_results['dy'])

        x = self._dmp_traj_gen.update_x(x, self._dmp_traj_gen.dt)
        self._dmp_traj_gen.set_x(x)

        self.step += 1

    def is_success(self, info):
        # x = self._dmp_traj_gen.get_x()
        # return x < 1e-4
        return False

    def _get_reach_pos(self, info):
        '''Get target position for the end-effector.
        
        If tool_offset is not None, then the tool offset is added to goal position, to ensure
        that the goal position is where the end-effector should be.
        '''
        goal_pos = self._target_keypoint_xyz
        if self._demo_offset_xyz is not None:
            goal_pos += self._demo_offset_xyz
        if self._tool_offset is not None:
            goal_pos += self._tool_offset

        return goal_pos

class DMPJointSkill(BaseSkill):
    def __init__(self, 
                 skill_type,
                 max_action_calls: int = 40,
                 params_key='',
                 use_axes=False,
                 axes=None,
                 dt=0.1,
                 add_weight_exploration=False,
                 **kwargs):
        super().__init__(
            skill_type=skill_type,
            use_gripper_params=False,
            use_ori_params=False,
            max_action_calls=max_action_calls,
            **kwargs,
        )
        self.add_weight_exploration = add_weight_exploration
        self.max_action_calls = max_action_calls
        self.params_key = params_key
        self.use_axes = use_axes
        self.axes = axes
        if not self.use_axes:
            self.axes = ['1', '2', '3', '4', '5', '6', '7']
        self.dt = dt
        self.rl_param_info = {'continuous': {'size':15, 'low':[], 'high':[], 'scale':False}, 'discrete':{'size':[]}}

    def get_param_dim(self, base_param_dim):
        return self.get_raw_param_dim()
    
    def get_raw_param_dim(self):
        return self.rl_param_info['continuous']['size']

    def log_parameters(self):
        '''Parameters to log for debugging what actions are being taken.'''
        size = self.get_raw_param_dim()
        if self.add_weight_exploration:
            values = self._params[:size].tolist()
        else:
            values = [0] * size
        names = [f'p_{i:02d}' for i in range(size)]
        return dict(values=values, size=size, names=names)

    def histogram_bins_to_log_parameters(self):
        num_params = self.get_raw_param_dim()
        low, high = -10 * np.ones(num_params,), 10 * np.ones(num_params)
        return np.split(np.linspace(low, high, 21), num_params, axis=1)
    
    def base_reset(self, config_update=None):
        # self._unscaled_params = params
        # self._target_keypoint_xyz = info['keypoint_xyz']
        # Demo offsets to use. If this is not none then the offset params are used as
        # residuals on top of these offsets.
        # self._demo_offset_xyz = info.get('demo_offset_xyz', None)
        # self._tool_offset = info['tool_offset'] if info.get('use_tool_offset', False) else None

        # valid_params = params[:self.get_raw_param_dim()]
        # self._params = np.copy(valid_params)

        # self._state = None
        # self._params_info = info
        # This will add robot_controller and similar vars to the config.
        if config_update is not None:
            self._config.update(config_update)
    
    def reset(self, params, start_pos, goal_pos, config_update=None):
        self.base_reset(config_update=config_update)

        # self.task_space, self.null_space = get_simple_projection_spaces(self.axes)
        self.task_space = np.eye(3)

        # dmp_params = info[self.params_key]
        self._dmp_traj_gen = DMPTrajectoryGenerator(
            params['tau'],
            params['alpha'],
            params['beta'],
            params['num_dims'],
            params['num_basis'],
            params['num_sensors'],
            add_min_jerk=params['add_min_jerk'],
            mu=params['mean'],
            gamma=params['gamma'],
            use_goal_formulation=params['use_goal_formulation'],
        )
        self.weights = np.copy(params['weights'])

        # Add params to weights
        # self.weight_params = np.zeros_like(self.weights)
        self.weight_params = np.copy(self.weights)
        # last_param_idx = 0
        # if 'x' in self.axes:
        #     self.weight_params[0, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        # if 'y' in self.axes:
        #     self.weight_params[1, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        # if 'z' in self.axes:
        #     self.weight_params[2, :, :] = self._params[last_param_idx:last_param_idx + self.weight_params.shape[2]]
        #     last_param_idx += self.weight_params.shape[2]
        
        self.goal_pos = goal_pos
        # self._goal_pos = self._get_reach_pos(info)
        # self.dt = 0.05 * 5
        self._dmp_traj_gen.init_with_weights(
            self.weights,
            # info['cur_ee_pos'],
            start_pos,
            self.dt,
            self.max_action_calls - 4,
            goal=self.goal_pos,
        )
        self.step = 0
    
    def get_pos_action(self, info):
        is_delta = False
        pos = np.array(self._dmp_traj_gen.y[self.step])

        # if task space projection makes something 0 (e.g. z) then its min bound may lie above.
        # if self.clip_xyz_bounds:
        #     pos = np.clip(pos, self.global_xyz_bounds[0, :], self.global_xyz_bounds[1, :])

        if self.use_axes:
            pos = self.task_space @ pos

        return pos, is_delta
    
    def get_ori_action(self, info):
        # ori = super().get_ori_ac({'params':[0,0,0,0,0,0,0]})
        ori = [0.0,0.0,0.0]
        # ori[:] = 0.0
        is_delta = True
        return ori, is_delta

    def get_gripper_action(self, info):
        return [0]
        # raise NotImplementedError

    def update_state(self, info):
        x = self._dmp_traj_gen.get_x()
        i = self.step

        # phi_ij would be any object based amplitude terms that can shape the trajectory
        # through the forcing function.
        dmp_step_results = self._dmp_traj_gen.step(
            x, i, self._dmp_traj_gen.y[i], self._dmp_traj_gen.dy[i], phi_ij=None)

        self._dmp_traj_gen.y.append(dmp_step_results['y'])
        self._dmp_traj_gen.dy.append(dmp_step_results['dy'])

        x = self._dmp_traj_gen.update_x(x, self._dmp_traj_gen.dt)
        self._dmp_traj_gen.set_x(x)

        self.step += 1

    def is_success(self, info):
        # x = self._dmp_traj_gen.get_x()
        # return x < 1e-4
        return False

    def _get_reach_pos(self, info):
        '''Get target position for the end-effector.
        
        If tool_offset is not None, then the tool offset is added to goal position, to ensure
        that the goal position is where the end-effector should be.
        '''
        goal_pos = self._target_keypoint_xyz
        if self._demo_offset_xyz is not None:
            goal_pos += self._demo_offset_xyz
        if self._tool_offset is not None:
            goal_pos += self._tool_offset

        return goal_pos