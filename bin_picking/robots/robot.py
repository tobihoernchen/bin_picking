from abc import ABC
from xml.etree import ElementTree as ET
from gymnasium import spaces
import numpy as np
import torch
from bin_picking.objects.mujoco_env import MujocoEnv
import mujoco
from bin_picking.objects.objects import Asset, XmlObject, XmlObjectCollection
import pytorch_kinematics as pk
from pytorch_kinematics.frame import Visual
import trimesh
import pathlib
import glob

BIN_PICKING_ROBOT_CACHE_FOLDER = pathlib.Path(__file__).parent / ".bin_picking/"


def mat_to_pos_quat(
    mat: torch.Tensor, device: torch.device, offset: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert transformation matrix to position and quaternion.

    Args:
        mat: Transformation matrix (torch.Tensor)
        device: PyTorch device
        offset: Offset tensor

    Returns:
        Tuple of (position_list, quaternion_list) for Mujoco compatibility
    """
    if isinstance(mat, np.ndarray):
        mat = torch.tensor(mat, dtype=torch.float32, device=device)
    elif isinstance(mat, torch.Tensor):
        mat = mat.to(device=device, dtype=torch.float32)

    # Extract position from homogeneous matrix
    pos = mat[..., :3, 3]
    pos = pos + offset if offset is not None else pos
    # Extract rotation matrix
    rot_mat = mat[..., :3, :3]
    # Convert to quaternion using pytorch_kinematics
    quat = pk.matrix_to_quaternion(rot_mat)

    # Return as lists for Mujoco compatibility
    return pos, quat


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


class Camera(XmlObject):
    def __init__(self, name, pos: str, euler: str, width: int = 480, height: int = 480):
        self.name = name
        self.env = None
        self.renderer = None
        self.width = width
        self.height = height
        super().__init__(
            "camera",
            {
                "name": name,
                "mode": "fixed",
                "pos": pos,
                "euler": euler,
            },
        )

    def initialize(self, env: MujocoEnv):
        self.env = env

    def render_image(self):
        m, d = self.env.get_mujoco()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, height=self.height, width=self.width)
        mujoco.mj_forward(m, d)
        self.renderer.update_scene(d, self.name)
        return self.renderer.render()

    def cleanup(self):
        """Clean up renderer resources"""
        if self.renderer is not None:
            self.renderer.close()


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


class Tool:
    def __init__(self):
        pass

    def add_to_chain(self, chain: pk.Chain, parent_frame: str):
        raise NotImplementedError("Tool must implement add_to_chain method")


class ParallelGripper(Tool):
    def __init__(self):
        pass

    def add_to_chain(self, chain: pk.Chain, parent_frame: pk.Frame):
        gripper_frame = pk.Frame("parallel_gripper")
        gripper_frame.joint = pk.Joint(type="fixed")
        gripper_frame.link = pk.Link(
            "parallel_gripper_link", None, []
        )  # No visuals for the gripper link
        parent_frame.children.append(gripper_frame)
        return gripper_frame


class PTPController:
    def __init__(
        self,
        env: MujocoEnv | None,
        chain: pk.Chain,
        axis_limits_rad: tuple[list[float]],
        axis_speed_rad_per_sec: tuple[list[float]],
        initial_axis_position: list[float] | None = None,
        device: str = "cpu",
    ):
        self.nbr_of_joints = len(axis_limits_rad[0])
        self.device = torch.device(device)
        self.chain = chain
        self.env = env
        # Convert to torch tensors for GPU-friendly operations
        self.axis_limits_rad = (
            torch.tensor(axis_limits_rad[0], dtype=torch.float32, device=self.device),
            torch.tensor(axis_limits_rad[1], dtype=torch.float32, device=self.device),
        )
        self.initial_axis_position = initial_axis_position

        self.axis_speed_rad_per_sec = (
            torch.tensor(axis_speed_rad_per_sec[0], dtype=torch.float32, device=self.device),
            torch.tensor(axis_speed_rad_per_sec[1], dtype=torch.float32, device=self.device),
        )
        self.reset()

    def forward_kinematics(
        self, axis_position: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        kinematics = self.chain.forward_kinematics(axis_position, end_only=False)
        return {
            frame: mat_to_pos_quat(kinematics[frame].get_matrix(), self.device)
            for frame in self.chain.get_frame_names()
        }

    def initialize(self, env: MujocoEnv):
        self.env = env

    def reset(self):
        self.axis_position = torch.tensor(
            self.initial_axis_position or [0.0] * self.nbr_of_joints,
            dtype=torch.float32,
            device=self.device,
        )
        self.link_position = None

        self.motion_actual_step = None
        self.motion_total_steps = None
        self.motion_link_positions = None
        self.motion_axis_endpoint = None
        self.motion_axis_positions = None
        self.in_motion = False

    def move_to(
        self, position: list[float] | torch.Tensor, clipping: bool = False, speed: float = 1.0
    ):

        # Convert to torch tensor if needed
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float32, device=self.device)
        else:
            position = position.to(device=self.device, dtype=torch.float32)

        if len(position) != self.nbr_of_joints:
            raise ValueError(f"Position must have {self.nbr_of_joints} elements")
        # check for axis limits using vectorized torch operations
        if clipping:
            position = torch.clamp(position, self.axis_limits_rad[0], self.axis_limits_rad[1])
        else:
            out_of_bounds = torch.logical_or(
                position < self.axis_limits_rad[0], position > self.axis_limits_rad[1]
            )
            if torch.any(out_of_bounds):
                idx = torch.where(out_of_bounds)[0][0].item()
                raise ValueError(
                    f"Position for joint {idx} must be between {self.axis_limits_rad[0][idx].item():.4f} and {self.axis_limits_rad[1][idx].item():.4f}"
                )
        if self.in_motion:
            self.terminate_motion()

        if self.env is None:
            raise ValueError("Environment must be set before moving")
        timestep = self.env.get_mujoco()[0].opt.timestep
        delta_per_axis = position - self.axis_position
        if delta_per_axis.abs().max() < 1e-6:
            return  # No movement needed
        speeds = torch.where(
            delta_per_axis < 0, self.axis_speed_rad_per_sec[0], self.axis_speed_rad_per_sec[1]
        )
        duration_per_axis = torch.abs(delta_per_axis / (speeds * speed))
        leading_axis = torch.argmax(duration_per_axis).item()
        motion_duration = duration_per_axis[leading_axis].item()
        motion_steps = int(motion_duration / timestep)

        axis_values_along_motion = self.axis_position.unsqueeze(0) + torch.linspace(
            0, 1, int(motion_steps) + 1, device=self.device
        ).unsqueeze(1) * delta_per_axis.unsqueeze(0)

        self.motion_actual_step = 0
        self.motion_total_steps = motion_steps
        self.motion_axis_endpoint = position
        self.motion_axis_positions = axis_values_along_motion
        self.motion_link_positions = self.forward_kinematics(axis_values_along_motion)
        self.in_motion = True

    def terminate_motion(self):
        if self.in_motion:
            self.get_link_positions(False)
        self.in_motion = False

    def get_link_positions(self, offset: torch.Tensor, go_to_next=False):
        if not self.in_motion and self.link_position is None:
            return {
                k: (v[0][0] + offset, v[1][0])
                for k, v in self.forward_kinematics(self.axis_position.unsqueeze(0)).items()
            }
        self.motion_actual_step += 1 if go_to_next else 0
        if self.motion_actual_step >= self.motion_total_steps:
            self.axis_position = self.motion_axis_positions[-1]
            self.link_position = {
                k: (v[0][-1] + offset, v[1][-1]) for k, v in self.motion_link_positions.items()
            }
            self.in_motion = False
            return self.link_position
        else:
            # Vectorized interpolation with torch tensors
            self.axis_position = self.motion_axis_positions[int(self.motion_actual_step)]
            self.link_position = {
                k: (v[0][int(self.motion_actual_step)] + offset, v[1][int(self.motion_actual_step)])
                for k, v in self.motion_link_positions.items()
            }
            return self.link_position


class ActiveMujocoComponent(ABC):
    def __init__(self):
        super().__init__()
        self.collection: XmlObjectCollection

    def initialize(self, env: MujocoEnv):
        raise NotImplementedError

    def reset(self):
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
        device: str = "cpu",
        timestep: float = 0.01,
        initial_axis_position: list[float] = None,
    ):
        self.chain = chain
        self.device = torch.device(device)
        self.timestep = timestep
        self.cameras: list[Camera] = []
        self.position = torch.zeros(3, device=self.device)

        self.kinematic_links: dict[str, KinematicLink] = {
            frame: None for frame in self.chain.get_frame_names()
        }
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
                    *self.get_pos_quat(link.offset.get_matrix()),
                )
            else:
                self.kinematic_links[link.name] = KinematicLink(
                    f"{name}_link_{i}",
                    geometry_name,
                    link.visuals,
                    *self.get_pos_quat(link.offset.get_matrix()),
                )

        limits = chain.get_joint_limits()
        velocities = chain.get_joint_velocity_limits()

        self.collection = XmlObjectCollection(
            list(self.kinematic_links.values()) + list(self.dead_links.values())
        )

        self.controller = PTPController(
            None,
            self.chain,
            axis_limits_rad=limits,
            axis_speed_rad_per_sec=velocities,
            initial_axis_position=initial_axis_position,
            device=device,
        )

        self.action_space = spaces.Box(
            low=np.array(limits[0], dtype=np.float32),
            high=np.array(limits[1], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self):
        self.controller.reset()
        # for camera in self.cameras:
        #     camera.reset()

    def get_pos_quat(self, mat: torch.Tensor) -> tuple[list[float], list[float]]:
        pos, quat = mat_to_pos_quat(mat, self.device)
        return pos.flatten().cpu().numpy().tolist(), quat.flatten().cpu().numpy().tolist()

    def to_position(self, position, clipping=False, speed=1.0):
        return self.controller.move_to(position, clipping=clipping, speed=speed)

    def ee_link(self):
        return self.kinematic_links[self.chain.get_frame_names()[-1]]

    def add_ee_camera(
        self,
        camera_name: str,
        pos: tuple[float, float, float],
        euler: tuple[float, float, float],
    ):
        camera = Camera(camera_name, " ".join(map(str, pos)), " ".join(map(str, euler)))
        self.cameras.append(camera)
        self.ee_link().append(camera)

    def initialize(self, env: MujocoEnv):
        self.controller.initialize(env)
        for camera in self.cameras:
            camera.initialize(env)

    def get_axis_value(self):
        return self.controller.get_axis_value()

    def get_link_positions(self):
        link_positions = self.controller.get_link_positions(offset=self.position, go_to_next=True)
        return {
            link.mocap_name: link_positions[joint_name]
            for joint_name, link in self.kinematic_links.items()
        }

    def move(self, dx, dy, dz):
        self.position += np.array([dx, dy, dz])
        for link in self.dead_links.values():
            link.move(dx, dy, dz)
