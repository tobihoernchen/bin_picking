from abc import ABC
from xml.etree import ElementTree as ET
from gymnasium import spaces
import numpy as np
from bin_picking.objects.mujoco_env import MujocoEnv
from bin_picking.objects.objects import Asset, XmlObject, XmlObjectCollection
import pytorch_kinematics as pk
from pytorch_kinematics.frame import Visual
import trimesh
import pathlib
import glob

BIN_PICKING_ROBOT_CACHE_FOLDER = pathlib.Path(__file__).parent / ".bin_picking/"


def get_simplified_mesh(path, face_count=200):
    mesh = trimesh.load_mesh(path)
    simpler_mesh = mesh.convex_hull
    simpler_mesh = simpler_mesh.simplify_quadric_decimation(
        face_count=min(face_count, len(simpler_mesh.faces)), aggression=3
    )
    return simpler_mesh


def register_robot_meshes_from_menagerie(menagerie_path: str, name: str):
    path = pathlib.Path(menagerie_path)
    obj_path = path / "assets"
    obj_files = glob.glob(str(obj_path / "*.obj"))

    decomposed_path = BIN_PICKING_ROBOT_CACHE_FOLDER / name
    decomposed_path.mkdir(parents=True, exist_ok=True)
    for file in obj_files:
        mesh = get_simplified_mesh(file)
        mesh.export(decomposed_path / pathlib.Path(file).name)


class AbstractLink(XmlObject):
    def get_mesh_geom(self, file_path: pathlib.Path, mesh_name: str, pos: str, quat: str):
        self.assets.add(
            Asset(
                "mesh",
                {
                    "name": mesh_name,
                    "file": str(file_path.resolve()),
                },
            )
        )
        geom = ET.Element(
            "geom",
            {
                "type": "mesh",
                "mesh": mesh_name,
                "material": "body_material",
                "friction": "1 0.005 0.0001",
                "pos": pos,
                "quat": quat,
            },
        )
        return geom

    def build_visual(
        self,
        robot_name,
        visual: Visual,
        pos: tuple[float, float, float] = (0, 0, 0),
        quat: tuple[float, float, float, float] = (0, 0, 0, 1),
    ):
        if visual.geom_type == "mesh":
            link_name = visual.geom_param[0].split("/")[-1].split(".")[0]
            obj_files = (BIN_PICKING_ROBOT_CACHE_FOLDER / robot_name).glob(f"{link_name}*.obj")
            geoms_elements = []
            for obj_file in obj_files:
                geoms_elements.append(
                    self.get_mesh_geom(
                        obj_file, obj_file.stem, " ".join(map(str, pos)), " ".join(map(str, quat))
                    )
                )
            return geoms_elements

        else:
            raise NotImplementedError(f"Unsupported geometry type: {visual.geom_type}")


class DeadLink(AbstractLink):
    def __init__(self, name, robot_name, visuals: list[Visual], pos: str, quat: str):
        super().__init__(
            "body",
            {"name": name},
        )
        geom_lists = [self.build_visual(robot_name, visual, pos, quat) for visual in visuals]
        for geom_list in geom_lists:
            for mesh_geom in geom_list:
                self.append(mesh_geom)

        material = Asset("material", {})
        material.set("name", "body_material")
        self.assets.add(material)


class KinematicLink(AbstractLink):
    def __init__(self, name, robot_name, visuals: list[Visual], pos: str, quat: str):
        self.mocap_name = f"{name}_mocap"
        super().__init__(
            "body",
            {"name": self.mocap_name, "mocap": "true"},
        )

        welded_body = XmlObject("body", {"name": f"{name}_visual"})
        geom_lists = [self.build_visual(robot_name, visual, pos, quat) for visual in visuals]
        for geom_list in geom_lists:
            for mesh_geom in geom_list:
                welded_body.append(mesh_geom)
        self.append(welded_body)

        material = Asset("material", {})
        material.set("name", "body_material")
        self.assets.add(material)


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

    def move_to(self, position: list[float], clipping: bool = False):
        if len(position) != self.nbr_of_joints:
            raise ValueError(f"Position must have {self.nbr_of_joints} elements")
        # check for axis limits
        if clipping:
            position = [
                max(
                    self.axis_limits_rad[0][i],
                    min(self.axis_limits_rad[1][i], position[i]),
                )
                for i in range(self.nbr_of_joints)
            ]
        else:
            for i in range(self.nbr_of_joints):
                if (
                    position[i] < self.axis_limits_rad[0][i]
                    or position[i] > self.axis_limits_rad[1][i]
                ):
                    raise ValueError(
                        f"Position for joint {i} must be between {self.axis_limits_rad[0][i]} and {self.axis_limits_rad[1][i]}"
                    )
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
        geometry_name: str,
    ):
        self.chain = chain

        self.kinematic_links = {frame: None for frame in self.chain.get_frame_names()}
        self.dead_links = {}

        for i, link in enumerate(self.chain.get_links()):
            if (
                link.offset is None
                or len(link.visuals) == 0
                or all([v.geom_type is None for v in link.visuals])
            ):
                continue
            if link.name not in self.kinematic_links:
                self.dead_links[link.name] = DeadLink(
                    f"{name}_link_{i}",
                    geometry_name,
                    link.visuals,
                    *self.mat_to_pos_quat(link.offset.get_matrix()),
                )
            else:
                self.kinematic_links[link.name] = KinematicLink(
                    f"{name}_link_{i}",
                    geometry_name,
                    link.visuals,
                    *self.mat_to_pos_quat(link.offset.get_matrix()),
                )

        limits = chain.get_joint_limits()
        velocities = chain.get_joint_velocity_limits()

        self.collection = XmlObjectCollection(
            list(self.kinematic_links.values()) + list(self.dead_links.values())
        )

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
            for joint_name, link in self.kinematic_links.items()
        }

    def mat_to_pos_quat(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = mat[..., :3, 3].flatten().tolist()
        rot_mat = mat[..., :3, :3]
        quat = pk.matrix_to_quaternion(rot_mat).flatten().tolist()
        return pos, quat
