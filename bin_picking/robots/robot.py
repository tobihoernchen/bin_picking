from typing import Callable
from xml.etree import ElementTree as ET
from gymnasium import spaces
import numpy as np
from bin_picking.objects.mujoco_env import MujocoEnv
from bin_picking.objects.objects import XmlObject, XmlObjectCollection


class KinematicLink(XmlObject):
    def __init__(
        self,
        name,
        alpha,
        link_length,
        offset_joint,
        theta,
        last_link: "KinematicLink" = None,
        is_last: bool = False,
    ):
        self.mocap_name = f"{name}_mocap"
        super().__init__(
            "body",
            {"name": self.mocap_name, "mocap": "true"},
        )
        self.position_is_calculated = False
        self.calculated_t_mat = None
        self.axis_position = None

        self.last_link = last_link
        self.rot_x = np.deg2rad(alpha)
        self.trans_x = link_length
        self.rot_z = np.deg2rad(theta)
        self.trans_z = offset_joint
        self.is_last = is_last
        self.welded_body = XmlObject("body", {"name": f"{name}_collision"})
        self.append(self.welded_body)
        self.welded_body.append(
            ET.Element(
                "geom",
                {
                    "type": "cylinder" if is_last or last_link is None else "capsule",
                    "fromto": f"0 0 0 {link_length} 0 {offset_joint}",
                    "size": "0.02",
                },
            )
        )

    def reset_position_calculation(self):
        self.position_is_calculated = False
        self.calculated_t_mat = None

    def set_axis_position(self, axis_position):
        self.axis_position = axis_position
        self.reset_position_calculation()

    def set_position_calculated(self, t_mat):
        self.position_is_calculated = True
        self.calculated_t_mat = t_mat

    def calculate_t_mat_recursively(self):
        if self.position_is_calculated:
            return self.calculated_t_mat
        if self.last_link is None:
            t_mat = np.eye(4)
            t_mat[0, 3] = self.trans_x
            t_mat[2, 3] = self.trans_z
            self.set_position_calculated(t_mat)
            return t_mat
        else:
            prev_t_mat = self.last_link.calculate_t_mat_recursively()
            t_mat = prev_t_mat @ self.get_own_t_mat(self.axis_position)
            self.set_position_calculated(t_mat)
            return t_mat

    def get_own_t_mat(self, axis_angle):
        axis_rad = np.deg2rad(axis_angle)
        return np.array(
            [
                [
                    np.cos(self.rot_z + axis_rad),
                    -np.sin(self.rot_z + axis_rad) * np.cos(self.rot_x),
                    np.sin(self.rot_z + axis_rad) * np.sin(self.rot_x),
                    self.trans_x * np.cos(self.rot_z + axis_rad),
                ],
                [
                    np.sin(self.rot_z + axis_rad),
                    np.cos(self.rot_z + axis_rad) * np.cos(self.rot_x),
                    -np.cos(self.rot_z + axis_rad) * np.sin(self.rot_x),
                    self.trans_x * np.sin(self.rot_z + axis_rad),
                ],
                [0, np.sin(self.rot_x), np.cos(self.rot_x), self.trans_z],
                [0, 0, 0, 1],
            ]
        )


class PTPMocapActor:
    def __init__(
        self,
        env: MujocoEnv | None,
        name,
        axis_limits_deg: list[tuple[float, float]],
        axis_speed_deg_per_sec: list[float],
        bodies: list[KinematicLink] = [],
        initial_axis_position=None,
    ):
        self.name = name
        self.bodies = bodies
        self.collection = XmlObjectCollection(bodies)
        self.nbr_of_joints = len(axis_limits_deg)
        self.axis_limits_deg = axis_limits_deg
        self.axis_position = initial_axis_position or [0.0] * self.nbr_of_joints
        if self.nbr_of_joints != len(axis_limits_deg):
            raise ValueError(
                "Length of axis_limits_deg and axis_speed_deg_per_sec must match nbr_of_joints"
            )
        self.action_space = spaces.Box(
            low=np.array([limit[0] for limit in axis_limits_deg], dtype=np.float32),
            high=np.array([limit[1] for limit in axis_limits_deg], dtype=np.float32),
            dtype=np.float32,
        )
        self.env = env
        self.callback_time = lambda: 0

        self.axis_speed_deg_per_sec = axis_speed_deg_per_sec
        self.motion_startpoint = None
        self.motion_starttime = None
        self.motion_endpoint = None
        self.motion_endtime = None
        self.in_motion = False
        if self.nbr_of_joints != len(axis_speed_deg_per_sec):
            raise ValueError(
                "Length of axis_limits_deg and axis_speed_deg_per_sec must match nbr_of_joints"
            )

    def initialize(self, env: MujocoEnv):
        self.env = env
        self.callback_time = lambda: 0 if self.env.d is None else self.env.d.time

    def move_to(self, position: list[float]):
        if len(position) != self.nbr_of_joints:
            raise ValueError(f"Position must have {self.nbr_of_joints} elements")
        if self.in_motion:
            self.terminate_motion()
        delta_per_axis = [
            abs(position[i] - self.axis_position[i]) for i in range(self.nbr_of_joints)
        ]
        leading_axis = max(range(self.nbr_of_joints), key=lambda i: delta_per_axis[i])
        motion_duration = delta_per_axis[leading_axis] / self.axis_speed_deg_per_sec[leading_axis]
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

    def get_link_positions(self):
        for body in self.bodies:
            body.reset_position_calculation()
        axis_positions = self.get_axis_value()
        for body, ap in zip(self.bodies, axis_positions):
            body.set_axis_position(ap)
        return {
            body.mocap_name: self.t_mat_to_pos_rot(body.calculate_t_mat_recursively())
            for body in self.bodies
        }

    def t_mat_to_pos_rot(self, t_mat):
        pos = t_mat[:3, 3]
        rot = np.array(
            [
                np.rad2deg(np.arctan2(t_mat[2, 1], t_mat[2, 2])),  # roll
                np.rad2deg(
                    np.arctan2(-t_mat[2, 0], np.sqrt(t_mat[2, 1] ** 2 + t_mat[2, 2] ** 2))
                ),  # pitch
                np.rad2deg(np.arctan2(t_mat[1, 0], t_mat[0, 0])),  # yaw
            ]
        )
        return pos, rot


class ParallelGripper(PTPMocapActor):
    def __init__(self, name):
        super().__init__(
            name,
            axis_limits_deg=[(0, 100)],
            axis_speed_deg_per_sec=[10],
            initial_axis_position=[0],
        )


class Kinematics6DOF(PTPMocapActor):
    def __init__(self, name):
        super().__init__(
            name,
            axis_limits_deg=[(-180, 180)] * 6,
            axis_speed_deg_per_sec=[30] * 6,
            initial_axis_position=[0] * 6,
        )


class Robot(PTPMocapActor):
    def __init__(
        self,
        name,
        tool: PTPMocapActor,
        neutral_dh_parameters: list[tuple[float, float, float, float]],
        axis_limits_deg: list[tuple[float, float]],
        axis_speed_deg_per_sec: list[float],
        initial_axis_position=None,
    ):
        super().__init__(
            None,
            name=name,
            callback_time=lambda: 0.0,
            axis_limits_deg=axis_limits_deg + tool.axis_limits_deg if tool else axis_limits_deg,
            axis_speed_deg_per_sec=axis_speed_deg_per_sec + tool.axis_speed_deg_per_sec
            if tool
            else axis_speed_deg_per_sec,
            initial_axis_position=initial_axis_position + tool.axis_position
            if initial_axis_position and tool
            else [0.0] * (len(axis_limits_deg)) + tool.axis_position
            if tool
            else initial_axis_position,
        )
        self.dh_parameters = neutral_dh_parameters
        self.links = []

        for i in range(len(axis_limits_deg) + 1):
            self.links.append(
                KinematicLink(
                    name=f"link_{i}",
                    alpha=neutral_dh_parameters[i][0],
                    link_length=neutral_dh_parameters[i][1],
                    theta=neutral_dh_parameters[i][2],
                    offset_joint=neutral_dh_parameters[i][3],
                    last_link=self.links[-1] if self.links else None,
                    is_last=(i == len(axis_limits_deg)),
                )
            )
        self.xml_objects = [link for link in self.links]

    def move_to(self, position):
        return super().move_to(position)

    def get_axis_value(self):
        return super().get_axis_value()
