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

    # print welcome info
    print("\n\nEntering dmp script:\n\n")

    # Choose environment and add it to options
    # options["env_name"] = choose_environment()
    options["env_name"] = 'Lift'

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    options["robots"] = 'Panda'

    # Choose controller
    controller_name = choose_controller()
    print(f"Chose controller {controller_name}")

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    options['control_freq'] = 1

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

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

    demo_path = "./robosuite/demos/data/demo_params_info.pkl"
    with open(demo_path, 'rb') as file:
        data = pickle.load(file)

    params = data['controller_info']['seg:00_dmp_xyz']['dmp_params']

    # look at lift.py
    # start_pos = [-0.10584563, 0.00258933, 1.00908577]
    # goal_pos = [-0.00563642, -0.00728302, 0.00751715]
    # print(params)
    # print(start_pos)
    # print(goal_pos)

    # print(params)

    # dmp = skill_dmp.DMPPositionSkill('lift')

    z_offset = 0.011
    start_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
    goal_pos = np.array(env.sim.data.body_xpos[env.cube_body_id])

    # goal position plus in Z direction
    goal_pos[2] = goal_pos[2] + z_offset
    sc.reset_dmp(params, start_pos, goal_pos)

    # do visualization
    for i in range(10000):
        action = sc.step_dmp()
        obs, reward, done, _ = env.step(action)
        env.render()
