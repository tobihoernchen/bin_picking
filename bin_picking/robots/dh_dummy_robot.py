from bin_picking.robots.robot import Kinematics, Robot, Joint
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoRenderer
from bin_picking.envs.mujocoenv import ImprovedRenderer


class DhDummyRobot(Robot):
    def __init__(self, width: int = 432, height: int = 240):
        self.width = width
        self.height = height
        self.renderer = None
        self.observation_space = gym.spaces.Dict(
            {
                "joints": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "ee_position": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "image": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(width, height), dtype=np.float32
                ),
            }
        )
        super().__init__(
            observation_space=self.observation_space,
            kinematics=Kinematics(
                Joint(
                    "1", (0, 0, 0.15), joint_range=(-175, 175), axis="Z", thickness=0.12
                ),
                Joint(
                    "2",
                    (0, 0.02025, 0.13787),
                    joint_range=(-90, 90),
                    axis="X",
                    thickness=0.1,
                ),
                Joint(
                    "3",
                    (0, 0, 0.26097),
                    joint_range=(-20, 110),
                    axis="X",
                    thickness=0.08,
                ),
                Joint(
                    "4", (0, 0, 0.130), joint_range=(-90, 90), axis="Z", thickness=0.07
                ),
                Joint(
                    "5", (0, 0, 0.130), joint_range=(-90, 90), axis="X", thickness=0.06
                ),
                Joint(
                    "6",
                    (0, 0, 0.07474),
                    joint_range=(-90, 90),
                    axis="Z",
                    thickness=0.055,
                ),
                Joint("end", (0, 0, 0.05), joint_type=None, thickness=0.05),
                base_thickness=0.12,
            ),
        )

    def register(self, sim):
        super().register(sim)
        self.renderer = ImprovedRenderer(
            self.sim_model,
            self.sim_data,
            width=self.width,
            height=self.height,
            camera_name="ee_cam",
        )

    def get_observation(self):
        return {
            "joints": self.sim_data.qpos[:7].copy(),
            "ee_position": self.sim_data.body("body_end").xpos.copy(),
            "image": self.get_camera_image(),
        }

    def get_camera_image(self):
        return self.renderer.render(render_mode="rgb_array")

    def take_action(self, action: np.ndarray):
        self.sim_data.ctrl[:6] = action[:6]
