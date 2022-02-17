from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


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

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
