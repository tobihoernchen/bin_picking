from typing import Literal, Optional
from gymnasium.spaces import Space
import mujoco
import numpy as np


class Joint:
    axis_vec = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}

    def __init__(
        self,
        name: str,
        pos: tuple[float, float, float],
        joint_type="hinge",
        joint_range: tuple[float, float] = (-180, 180),
        axis: Literal["X", "Y", "Z"] = "Z",
        thickness=None,
    ):
        self.name = name
        self.pos = np.array(pos)
        self.joint_type = joint_type
        self.joint_range = joint_range
        self.axis = axis
        self.thickness = thickness

    def body_to_xml(self, follower: Optional["Joint"] = None, indent_lvl=0) -> list[str]:
        length = np.linalg.norm(follower.pos) if follower is not None else 0.03
        thickness = self.thickness if self.thickness is not None else length / 3
        lines = [
            "  " * indent_lvl + f'<body name="body_{self.name}" pos="{self.pos_string()}">',
            "  " * (indent_lvl + 1)
            + f'<geom name="geom_{self.name}" type="{"capsule" if self.joint_type is not None else "cylinder"}" fromto="0 0 0 {follower.pos_string() if follower is not None else "0 0 0.1"}" size="{thickness}"/>',
        ]
        if self.joint_type == "hinge" and self.axis == "X":
            lines.append(
                "  " * (indent_lvl + 1)
                + f'<geom name="geom_{self.name}_2" type="cylinder" fromto="{-thickness} 0 0 {thickness} 0  0" size="{thickness}"/>',
            )
        if self.joint_type is not None:
            lines.append(
                "  " * (indent_lvl + 1)
                + f'<joint name="joint_{self.name}" type="{self.joint_type}" pos="0 0 0" axis="{self.axis_vec[self.axis]}" range="{self.joint_range[0]} {self.joint_range[1]}" damping="1.0"/>'
            )
        return lines

    def pos_string(self):
        return " ".join(map(str, self.pos))

    def actuator_to_xml(self, indent_lvl=0) -> str:
        return (
            "  " * indent_lvl
            + f'<position joint="joint_{self.name}" ctrlrange="{self.joint_range[0]} {self.joint_range[1]}" ctrllimited="true" />'
        )


class Kinematics:
    def __init__(self, *joints: Joint, base_thickness=0.12):
        self.joints = joints
        self.base_thickness = base_thickness

    def body_to_xml(self, indent_lvl=2):
        thickness = (
            np.linalg.norm(self.joints[0].pos) / 2
            if self.base_thickness is None
            else self.base_thickness
        )
        lines = [
            "  " * indent_lvl + '<body name="body_base" pos="0 0 0">',
            "  " * (indent_lvl + 1)
            + f'<geom name="geom_base" type="cylinder" fromto="0 0 0 {self.joints[0].pos_string()}" size="{thickness}"/>',
        ]
        closing = ["  " * indent_lvl + "</body>"]
        for joint, follower in zip(
            self.joints,
            self.joints[1:] + (None,),
        ):
            indent_lvl += 1
            lines.extend(joint.body_to_xml(follower=follower, indent_lvl=indent_lvl))
            closing.append("  " * indent_lvl + "</body>")
        closing.reverse()
        return "\n".join(lines + closing)

    def actuator_to_xml(self, indent_lvl=2):
        return "\n".join(
            joint.actuator_to_xml(indent_lvl=indent_lvl)
            for joint in self.joints
            if joint.joint_type is not None
        )


class Robot:
    def __init__(self, observation_space: Space, kinematics: Kinematics = None):
        self.observation_space = observation_space
        self.sim_model, self.sim_data = None, None
        self.kinematics = kinematics

    def register(self, sim: tuple["mujoco.MjModel", "mujoco.MjData"]):
        self.sim_model, self.sim_data = sim

    def get_observation(self):
        raise NotImplementedError

    def take_action(self, action: np.ndarray):
        raise NotImplementedError

    def add_to_spec(self, spec):
        return spec

    def robot_body_xml(self):
        if self.kinematics is None:
            raise NotImplementedError
        return self.kinematics.body_to_xml()

    def robot_actuators_xml(self):
        if self.kinematics is None:
            raise NotImplementedError
        return self.kinematics.actuator_to_xml()
