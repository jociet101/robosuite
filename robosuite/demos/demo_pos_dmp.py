import numpy as np
import pickle
import pdb

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.controllers.skill_controller import SkillController
# from robosuite.environments.robot_env import RobotEnv

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    options["env_name"] = 'Lift'

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    options["robots"] = 'Panda'

    # Choose controller
    controller_name = 'OSC_POSE'

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    options['control_freq'] = 2

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # load weight data from pkl file for both segments
    demo_path = "./robosuite/demos/data/trial.pkl"
    with open(demo_path, 'rb') as file:
        data = pickle.load(file)

    params1 = data['controller_info']['seg:00_dmp_xyz']['dmp_params']
    params2 = data['controller_info']['seg:01_dmp_xyz']['dmp_params']

    param_info_dict = {'continuous': {'size': 15, 'low': [], 'high': [], 'scale': False},
            'discrete': {'size': []}}

    config = {'skills': ['dmp'],
        'max_action_calls': 40,
        'params_key': 'stage_0_dmp_xyz',
        'use_axes': True,
        'scale_bounds': {'low': [], 'high': []},
        'param_info': param_info_dict}

    sc = SkillController(None, config)

    options['skill_controller'] = sc
    options['skill_name'] = 'dmp'

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        # control_freq=20
        # skill_controller=sc
    )
    # env = RobotEnv()
    env.reset()
    env.viewer.set_camera(camera_id=0)
    sc.env = env

    # Get action limits
    low, high = env.action_spec

    # for segment 1
    # z_offset = 0.028
    start_pos1 = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
    goal_pos1 = np.array(env.sim.data.body_xpos[env.cube_body_id])

    # goal position plus in Z direction
    # goal_pos1[2] = goal_pos1[2] + z_offset
    # set dmp params
    sc.reset_dmp(params1, start_pos1, goal_pos1)

    def get_joint_values():
        # seg1_joint_names = env.robots[0].robot_joints
        joint_pos_idxs = env.robots[0]._ref_joint_pos_indexes
        joint_vel_idxs = env.robots[0]._ref_joint_vel_indexes

        eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
        joint_pos = np.array([env.sim.data.qpos[x] for x in joint_pos_idxs])
        joint_vel = np.array([env.sim.data.qvel[x] for x in joint_vel_idxs])

        dt = 0.01

        return eef_pos, dt, joint_pos, joint_vel

    segment1 = {'start_pos': start_pos1,
                'goal_pos':goal_pos1,
                'eef_pos':[],
                'dt':[],
                'joint_pos':np.zeros([20,7]),
                'joint_vel':np.zeros([20,7])}

    # do visualization
    for i in range(20):
        gripper_open = True
        if i > 15:
            gripper_open = False

        action = sc.step_dmp(gripper_open)
        
        obs, reward, done, _ = env.step(action, gripper_open)

        eef_pos, dt, joint_pos, joint_vel = get_joint_values()

        segment1['eef_pos'] += [eef_pos]
        segment1['dt'] += [dt]
        segment1['joint_pos'][i] = joint_pos
        segment1['joint_vel'][i] = joint_vel

        env.render()

    # import pdb; pdb.set_trace()

    print('segment 1 done')

    # for segment 2
    start_pos2 = np.copy(goal_pos1)
    goal_pos2 = np.copy(start_pos1)

    # set dmp params
    sc.reset_dmp(params2, start_pos2, goal_pos2)

    segment2 = {'start_pos': start_pos2,
                'goal_pos':goal_pos2,
                'eef_pos':[],
                'dt':[],
                'joint_pos':np.zeros([20,7]),
                'joint_vel':np.zeros([20,7])}

    # do visualization
    for i in range(20):
        gripper_open = False

        action = sc.step_dmp(gripper_open)
        obs, reward, done, _ = env.step(action, gripper_open)

        eef_pos, dt, joint_pos, joint_vel = get_joint_values()

        segment2['eef_pos'] += [eef_pos]
        segment2['dt'] += [dt]
        segment2['joint_pos'][i] = joint_pos
        segment2['joint_vel'][i] = joint_vel

        env.render()

    print('segment 2 done')
    
    # save the segmented data into info dictionary, then into pkl file
    info = {0: segment1, 1: segment2}

    import os
    filename = 'data/joint_dmp_data.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(info, f)
