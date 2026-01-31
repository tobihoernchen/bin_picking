from os import path
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class CameraReachEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "../bin_picking/robots/dummy.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_target: float = 1,
        reward_control_weight: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_target,
            reward_control_weight,
            **kwargs,
        )
        self._reward_dist_target = reward_dist_target
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        print(path.abspath("../bin_picking/tasks/camera_reach/dummy.xml"))
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_target
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        reward = reward_dist + reward_ctrl

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
        }

        return reward, reward_info

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                self.get_body_com("body_6"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
