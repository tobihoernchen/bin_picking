from abc import ABC
from xml.etree import ElementTree as ET
from gymnasium import spaces
import numpy as np
from bin_picking.objects.mujoco_env import MujocoEnv
from bin_picking.objects.objects import XmlObject, XmlObjectCollection
import pytorch_kinematics as pk


class Base(XmlObject):
    def __init__(self, name):
        super().__init__(
            "body",
            {"name": name},
        )
        self.append(
            ET.Element(
                "geom",
                {
                    "type": "cylinder",
                    "fromto": "0 0 0 0 0 0.1",
                    "size": "0.02",
                },
            )
        )


class KinematicLink(XmlObject):
    def __init__(
        self,
        name,
        sizes: tuple[float, float, float] = (0, 0, 0.1),
        geom_type="capsule",
    ):
        self.mocap_name = f"{name}_mocap"
        super().__init__(
            "body",
            {"name": self.mocap_name, "mocap": "true"},
        )

        self.welded_body = XmlObject("body", {"name": f"{name}_collision"})
        self.append(self.welded_body)
        self.welded_body.append(
            ET.Element(
                "geom",
                {
                    "type": geom_type,
                    "fromto": f"0 0 0 {-sizes[0]} {-sizes[1]} {-sizes[2]}",
                    "size": "0.02",
                },
            )
        )


class PTPController:
    def __init__(
        self,
        env: MujocoEnv | None,
        name,
        axis_limits_rad: tuple[list[float]],
        axis_speed_rad_per_sec: tuple[list[float]],
        initial_axis_position: list[float] | None = None,
    ):
        self.name = name
        self.nbr_of_joints = len(axis_limits_rad[0])
        self.axis_limits_rad = axis_limits_rad
        self.axis_position = initial_axis_position or [0.0] * self.nbr_of_joints
        if self.nbr_of_joints != len(axis_speed_rad_per_sec[0]):
            raise ValueError(
                "Length of axis_limits_rad and axis_speed_rad_per_sec must match nbr_of_joints"
            )
        self.env = env
        self.callback_time = lambda: 0

        self.axis_speed_rad_per_sec = axis_speed_rad_per_sec
        self.motion_startpoint = None
        self.motion_starttime = None
        self.motion_endpoint = None
        self.motion_endtime = None
        self.in_motion = False

    def initialize(self, env: MujocoEnv):
        self.env = env
        self.callback_time = lambda: 0 if self.env.d is None else self.env.d.time

    def move_to(self, position: list[float]):
        if len(position) != self.nbr_of_joints:
            raise ValueError(f"Position must have {self.nbr_of_joints} elements")
        if self.in_motion:
            self.terminate_motion()
        delta_per_axis = [position[i] - self.axis_position[i] for i in range(self.nbr_of_joints)]
        duration_per_axis = [
            delta_per_axis[i] / self.axis_speed_rad_per_sec[0 if delta_per_axis[i] < 0 else 1][i]
            for i in range(self.nbr_of_joints)
        ]
        leading_axis = max(range(self.nbr_of_joints), key=lambda i: duration_per_axis[i])
        motion_duration = duration_per_axis[leading_axis]
        self.motion_startpoint = self.axis_position.copy()
        self.motion_starttime = self.callback_time()
        self.motion_endpoint = position
        self.motion_endtime = self.motion_starttime + motion_duration
        self.in_motion = True

    def terminate_motion(self):
        self.axis_position = self.get_axis_value()
        self.in_motion = False

    def get_axis_value(self):
        if not self.in_motion:
            return self.axis_position
        current_time = self.callback_time()
        if current_time >= self.motion_endtime:
            self.axis_position = self.motion_endpoint
            self.in_motion = False
            return self.axis_position
        else:
            progress = (current_time - self.motion_starttime) / (
                self.motion_endtime - self.motion_starttime
            )
            self.axis_position = [
                self.motion_startpoint[i]
                + progress * (self.motion_endpoint[i] - self.motion_startpoint[i])
                for i in range(self.nbr_of_joints)
            ]
            return self.axis_position


class ActiveMujocoComponent(ABC):
    def __init__(self):
        super().__init__()
        self.collection: XmlObjectCollection

    def initialize(self, env: MujocoEnv):
        raise NotImplementedError

    def get_link_positions(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Should return a dictionary mapping link names to their current position and orientation (as a quaternion).
        """
        raise NotImplementedError


class Robot(ActiveMujocoComponent):
    def __init__(
        self,
        name,
        chain: pk.Chain,
    ):
        self.chain = chain

        limits = chain.get_joint_limits()
        velocities = chain.get_joint_velocity_limits()
        joints = list(chain.get_joints())
        self.links = [
            KinematicLink(f"{name}_link_{i}", joint.offset.get_matrix()[..., :3, 3].flatten())
            for i, joint in enumerate(joints)
        ]
        self.bodies = [Base(f"{name}_base")] + list(self.links)
        self.collection = XmlObjectCollection(self.bodies)

        joints_to_links = {
            joint.name: links[0].name for joint, links in self.chain.get_joints_and_child_links()
        }
        self.joint_names_for_bodies = [joints_to_links[joint.name] for joint in joints]

        self.controller = PTPController(
            None,
            name=f"{name}_controller",
            axis_limits_rad=limits,
            axis_speed_rad_per_sec=velocities,
        )

        self.action_space = spaces.Box(
            low=np.array(limits[0], dtype=np.float32),
            high=np.array(limits[1], dtype=np.float32),
            dtype=np.float32,
        )

    def move_to(self, position):
        return self.controller.move_to(position)

    def initialize(self, env: MujocoEnv):
        self.controller.initialize(env)

    def get_axis_value(self):
        return self.controller.get_axis_value()

    def get_link_positions(self):
        axis_values = self.controller.get_axis_value()
        link_positions = self.chain.forward_kinematics(axis_values, end_only=False)
        return {
            link.mocap_name: self.mat_to_pos_quat(link_positions[joint_name].get_matrix())
            for link, joint_name in zip(self.links, self.joint_names_for_bodies)
        }

    def mat_to_pos_quat(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = mat[..., :3, 3].flatten().tolist()
        rot_mat = mat[..., :3, :3]
        quat = pk.matrix_to_quaternion(rot_mat).flatten().tolist()
        return pos, quat
