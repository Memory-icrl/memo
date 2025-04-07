import random
import warnings

try:
    import memory_maze
except:
    warnings.warn(
        "Missing mem maze Install: `pip install amago[envs]` or `pip install git+`",
        category=Warning,
    )
import gymnasium as gym
import numpy as np

from amago.envs.env_utils import space_convert
from amago.envs.builtin.gym_envs import GymEnv
from memory_maze import tasks
# from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from memory_maze.gym_wrappers import GymWrapper, KeepKeysGymWrapper


def get_maze_obs_space():
    obs_keys = ['image']  # , 'agent_pos', 'agent_dir' 
    all_obs_space = {
        'image': gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.float32),                 ## OR keep as int?
        'agent_dir': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'agent_pos': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),                                     ## TODO: reset later
        'maze_layout': gym.spaces.Box(0, 1, (9, 9), np.uint8),
        'target_color': gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        'target_index': gym.spaces.Box(-9223372036854775806, 9223372036854775805, (), np.int64),
        'target_memory': gym.spaces.Box(0, 255, (10, 3, 64, 64), np.uint8),
        'target_pos': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'target_vec': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'targets_in_view': gym.spaces.Box(0, 1, (3,), np.uint8),
        'targets_pos': gym.spaces.Box(-np.inf, np.inf, (3, 2), np.float32),
        'targets_vec': gym.spaces.Box(-np.inf, np.inf, (3, 2), np.float32),
        'target_time_index':  gym.spaces.Box(0, 4000, (10,), np.int64),
        'target_coords': gym.spaces.Box(-np.inf, np.inf, (10, 2), np.float32),  ## nake same as agent/-pos
        'absolute_position': gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        'absolute_orientation': gym.spaces.Box(-np.inf, np.inf, (3, 3), np.float32),
        # 'aligned_path_images': gym.spaces.Box(0, 255, (10, 3, 64, 64), np.uint8),
    }
    obs_space = gym.spaces.Dict({k: all_obs_space[k] for k in obs_keys})
    return obs_space

class MemMaze(GymEnv):
    
    def __init__(self, maze_size, keep_keys=['image', 'prev_action']): # 'agent_pos', 'agent_dir',
        self.maze_dict = {
            9: (tasks.memory_maze_9x9, 1000),
            11: (tasks.memory_maze_11x11, 2000),
            13: (tasks.memory_maze_13x13, 3000),
            15: (tasks.memory_maze_15x15, 4000),
        }
        # env = DmControlCompatibilityV0(
        #     self.maze_dict[
        #         self.maze_size
        #             ][0](
        #                 # image_only_obs=True
        #                 global_observables=True,
        #                 ), render_mode='rgb_array'
        #     )
        self.maze_size = maze_size
        env = KeepKeysGymWrapper(self.maze_dict[
                self.maze_size
                    ][0](
                        # image_only_obs=True
                        global_observables=True,
                        prev_action_wrapper=True,
                        # distance2goalreward=True,
                        # move_towards_goal_reward=0.005,
                        ),
            keep_keys=keep_keys            
            )
        self.horizon = self.maze_dict[maze_size][1]
        # env = self.maze_dict[
        #         self.maze_size
        #             ][0](
        #                 # image_only_obs=True
        #                 global_observables=True,
        #                 )
                        #, render_mode='rgb_array')
        # print(env.observation_space)

        super().__init__(
            gym_env=env,
            env_name=f"mem_maze",
            horizon=self.maze_dict[self.maze_size][1],
            start=0,
            zero_shot=True,
            convert_from_old_gym=True,
        )

if __name__ == "__main__":
    env = MemMaze(keep_keys=['image', 'agent_pos', 'agent_dir'])
    obs_space = env.observation_space 
    import ipdb
    ipdb.set_trace()
    compare_space = get_maze_obs_space()

    assert space_convert(obs_space) == space_convert(compare_space)

    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
        env.render()
    env.close()