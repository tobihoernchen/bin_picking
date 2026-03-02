from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET
import numpy as np
import trimesh
import mujoco
import mujoco.viewer
import time
from bin_picking.objects.objects import XmlObject, XmlObjectCollection
from bin_picking.objects.stl_objects import StlBody

if TYPE_CHECKING:
    from bin_picking.robots.robot import MujocoMocapActor


class MujocoEnv(XmlObject):
    def __init__(self, name: str = "default_model", objects: list[XmlObject] = []):
        super().__init__("mujoco", {"model": name})
        self.m: None | mujoco.MjModel = None
        self.d: None | mujoco.MjData = None
        self.worldbody = XmlObject("worldbody")
        self.components_active_at_runtime: list["MujocoMocapActor"] = []
        # Add Floor
        self.worldbody.append(
            ET.Element(
                "geom",
                {
                    "name": "floor",
                    "type": "plane",
                    "size": "5 5 0.1",
                    "rgba": "0.2 0.3 0.4 1",
                },
            )
        )
        self.append(self.worldbody)
        for obj in objects:
            self.worldbody.append(obj)

    def append_active_component(self, component: "MujocoMocapActor"):
        self.components_active_at_runtime.append(component)
        component.initialize(self)
        self.append_object_collection(component)

    def xml_spec(self) -> str:
        tree = ET.fromstring(ET.tostring(self, encoding="unicode"))
        tree.append(ET.fromstring(self.assets_to_xml()))
        ET.indent(tree)
        return ET.tostring(tree, encoding="unicode")

    def get_mesh_in_world(self, object: StlBody, data_object) -> trimesh.Trimesh:
        mesh = object.get_trimesh()
        name = object.attrib["name"]
        pos = data_object.body(name).xpos
        mat = data_object.body(name).xmat

        t_mat = np.eye(4)
        t_mat[:3, :3] = mat.reshape(3, 3)
        t_mat[:3, 3] = pos

        mesh.apply_transform(t_mat)
        return mesh

    def stls_in_world(self) -> trimesh.Scene:
        if self.m is None or self.d is None:
            raise ValueError("Mujoco model and data must be initialized first.")
        stl_objects = self.get_stl_objects()
        meshes_in_world = [self.get_mesh_in_world(o, self.d) for o in stl_objects]
        return trimesh.Scene(meshes_in_world)

    def append_object_collection(self, collection: XmlObjectCollection):
        for obj in collection.xml_objects:
            self.worldbody.append(obj)

    def get_mujoco(self):
        if self.m is not None and self.d is not None:
            return self.m, self.d
        xml_string = self.xml_spec()
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        self.m = model
        self.d = data
        return model, data

    def step(self, m, d):
        for component in self.components_active_at_runtime:
            component.update_simulation(self)
        mujoco.mj_step(m, d)

    def run_with_passive_viewer(self, duration_seconds: float = 30.0):
        m, d = self.get_mujoco()
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < duration_seconds:
                step_start = time.time()
                self.step(m, d)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                    viewer.sync()

                elapsed = time.time() - step_start
                if m.opt.timestep > elapsed:
                    time.sleep(m.opt.timestep - elapsed)

    def run_unrendered(self, duration_seconds: float = 30.0):
        m, d = self.get_mujoco()
        start = d.time
        while d.time < start + duration_seconds:
            self.step(m, d)
