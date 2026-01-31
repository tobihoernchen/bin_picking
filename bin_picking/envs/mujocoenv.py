from gymnasium.envs.mujoco.mujoco_env import MujocoEnv as GymMujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
from gymnasium.spaces import Space
import mujoco

from bin_picking.robots.robot import Robot
from bin_picking.tasks.task import Task


DEFAULT_SIZE = 480


class ImprovedRenderer(MujocoRenderer):
    def _get_viewer(self, render_mode: str):
        viewer = super()._get_viewer(render_mode)
        viewer.make_context_current()
        return viewer


class MujocoEnv(GymMujocoEnv):
    def __init__(
        self,
        task_type: type[Task],
        robot_type: type[Robot],
        frame_skip: int,
        render_mode: str | None = None,
        robot_kwargs: dict | None = None,
        task_kwargs: dict | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict[str, float | int] | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        self.width = width
        self.height = height

        self.robot = robot_type(**(robot_kwargs or {}))
        self.task = task_type(**(task_kwargs or {}))
        self.observation_space = self.robot.observation_space

        self.model, self.data = self._initialize_simulation()
        self.robot.register((self.model, self.data))
        self.task.register((self.model, self.data))

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        if "render_fps" in self.metadata:
            assert int(np.round(1.0 / self.dt)) == self.metadata["render_fps"], (
                f"Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata['render_fps']}"
            )
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.mujoco_renderer = ImprovedRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            camera_name,
            visual_options,
        )

    def reset(self, *, seed=None, options=None):
        self._initialize_simulation()
        observation = self.robot.get_observation()
        return observation, {}

    def step(self, action: np.ndarray):
        self.robot.take_action(action)

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self.robot.get_observation()
        reward = self.task.get_reward(action)
        return observation, reward, False, False, {}

    def _initialize_simulation(
        self,
    ) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        spec = mujoco.MjSpec.from_string(self.make_xml())
        spec = self.robot.add_to_spec(spec)
        spec = self.task.add_to_spec(spec)
        model = spec.compile()
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def make_xml(self):
        return (
            """
    
<mujoco model="arm3d">
  <compiler inertiafromgeom="true" angle="degree" coordinate="local"/>
  <option timestep="0.01"  iterations="20" integrator="Euler" />

  <default>
    <joint armature='0.04' damping="1" limited="true"/>
    <geom friction=".8 .1 .1" density="300" margin="0.002" condim="1" contype="1" conaffinity="1"/>
  </default>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    {{ROBOT_BODY}}
    {{TASK_BODY}}
  </worldbody>

  <actuator>
    {{ROBOT_ACTUATORS}}
    {{TASK_ACTUATORS}}
  </actuator>
</mujoco>

""".replace("{{ROBOT_BODY}}", self.robot.robot_body_xml())
            .replace("{{ROBOT_ACTUATORS}}", self.robot.robot_actuators_xml())
            .replace("{{TASK_BODY}}", self.task.task_body_xml())
            .replace("{{TASK_ACTUATORS}}", self.task.task_actuators_xml())
        )
