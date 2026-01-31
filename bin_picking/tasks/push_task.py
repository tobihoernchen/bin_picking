import numpy as np
from bin_picking.tasks.task import Task
import mujoco


class PushTask(Task):
    def __init__(
        self, reward_near_weight=1, reward_dist_weight=1, reward_control_weight=0.1
    ):
        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        super().__init__()

    def get_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        vec_1 = self.position_of("object") - self.position_of("body_end")
        vec_2 = self.position_of("object") - self.position_of("goal")

        reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        reward = reward_dist + reward_ctrl + reward_near

        return reward

    def add_to_spec(self, spec):
        goal_pos = np.concatenate(
            [
                np.random.uniform(low=0.1, high=0.4, size=1),
                np.random.uniform(low=0.1, high=0.4, size=1),
                np.array([0]),
            ]
        )
        while True:
            cylinder_pos = np.concatenate(
                [
                    np.random.uniform(low=0.1, high=0.4, size=1),
                    np.random.uniform(low=0.1, high=0.4, size=1),
                    np.array([0.1]),
                ]
            )
            if np.linalg.norm(cylinder_pos[:2] - goal_pos[:2]) > 0.17:
                break
        object = spec.worldbody.add_body(name="object", pos=cylinder_pos)
        object.add_geom(
            #            rgba=[1, 1, 1, 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.05, 0.05, 0.05],
        )
        object.add_joint(
            name="obj_slidey",
            type=mujoco.mjtJoint.mjJNT_FREE,
            pos=[0, 0, 0],
            axis=[0, 1, 0],
            range=[-10.3213, 10.3],
            damping=0.5,
        )

        goal = spec.worldbody.add_body(name="goal", pos=goal_pos)
        goal.add_geom(
            rgba=[1, 0, 0, 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.08, 0.001, 0.1],
            density=0.00001,
            contype=0,
            conaffinity=0,
        )
        goal.add_joint(
            name="goal_slidey",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            pos=[0, 0, 0],
            axis=[0, 1, 0],
            range=[-10.3213, 10.3],
            damping=0.5,
        )
        goal.add_joint(
            name="goal_slidex",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            pos=[0, 0, 0],
            axis=[1, 0, 0],
            range=[-10.3213, 10.3],
            damping=0.5,
        )
        return super().add_to_spec(spec)

    def task_body_xml(self):
        return """
    <geom name="table" type="plane" pos="0 0.5 0" size="1 1 0.1" contype="1" conaffinity="1"/>

    """

    def task_actuators_xml(self):
        return ""
