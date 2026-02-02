from xml.etree import ElementTree as ET
from abc import ABC
import pathlib
import numpy as np
import trimesh
import copy
import mujoco
import mujoco.viewer
import time

BIN_PICKING_CACHE_FOLDER = pathlib.Path(__file__).parent / ".bin_picking/"
TEXTURE_FOLDER = pathlib.Path(__file__).parent / "textures"


class XmlObject(ET.Element):
    def __init__(self, tag, attrib={}, **extra):
        self.assets = set()
        super().__init__(tag, attrib, **extra)

    def to_xml(self) -> str:
        ET.indent(self)
        return ET.tostring(self, encoding="unicode")

    def get_assets(self):
        assets = self.assets.copy()
        for child in self:
            if isinstance(child, XmlObject):
                assets = assets.union(child.get_assets())
        return assets

    def assets_to_xml(self) -> str:
        asset_element = ET.Element("asset")
        for asset in list(self.get_assets()):
            asset_element.append(asset)
        ET.indent(asset_element)
        return ET.tostring(asset_element, encoding="unicode")

    def at(self, x, y, z):
        if not self.tag in ["body", "geom", "site", "camera", "light"]:
            raise ValueError(f"Cannot set position for tag '{self.tag}'")
        self.set("pos", f"{x} {y} {z}")
        return self

    def rotate(self, x, y, z):
        if not self.tag in ["body", "geom", "site", "camera", "light"]:
            raise ValueError(f"Cannot set orientation for tag '{self.tag}'")
        self.set("euler", f"{x} {y} {z}")
        return self

    def get_stl_objects(self) -> list[str]:
        stl_objects = []
        for child in self:
            if hasattr(child, "get_trimesh"):
                stl_objects.append(child)
            elif isinstance(child, XmlObject):
                stl_objects.extend(child.get_stl_objects())
        return stl_objects


class Asset(XmlObject):
    def __init__(self, tag, attrib=..., **extra):
        super().__init__(tag, attrib, **extra)

    def __hash__(self):
        xml_str = ET.tostring(self, encoding="unicode")
        normalized = xml_str.replace("\n", "").replace(" ", "")
        return hash(normalized)

    def __eq__(self, value):
        return self.__hash__() == value.__hash__()


def register_stl_body(name: str, file_path, recreate=False, max_convex_hull=10) -> None:
    if not pathlib.Path(file_path).exists():
        raise FileNotFoundError(f"STL file '{file_path}' does not exist.")
    decomposed_path = BIN_PICKING_CACHE_FOLDER / name
    if decomposed_path.exists() and not recreate:
        print(f"STL body '{name}' is already skipped.")
        return
    mesh = trimesh.load_mesh(file_path)
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(trimesh.transformations.scale_and_translate(0.001))
    parts = mesh.convex_decomposition(maxConvexHulls=max_convex_hull)
    # mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
    # parts_coacd = coacd.run_coacd(
    #     mesh_coacd, threshold=0.2, max_convex_hull=max_convex_hull
    # )
    # parts = [trimesh.Trimesh(vertices=part.vertices, faces=part.faces) for part in parts_coacd]
    decomposed_path.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        part.export(decomposed_path / f"{name}_part_{i}.stl")
    return parts


class StlBody(XmlObject):
    COUNTER = 0

    def __init__(
        self, name, attrib={}, fixed=False, texture_type: str | None = None, **extra
    ):
        attrib["name"] = f"{name}_{self.COUNTER}"
        super().__init__("body", attrib, **extra)
        self.texture = None
        decomposed_path = BIN_PICKING_CACHE_FOLDER / name
        if texture_type is not None:
            self._add_texture(texture_type)
        parts = [part for part in decomposed_path.glob(f"{name}_part_*.stl")]
        if len(parts) == 0:
            raise FileNotFoundError(f"STL body '{name}' has no parts.")
        for i, part in enumerate(parts):
            geom = ET.SubElement(
                self,
                "geom",
                {
                    "type": "mesh",
                    "mesh": f"{name}_part_{i}",
                    "material": "body_material"
                    if not self.texture
                    else f"{self.texture}_material",
                    "friction": "1 0.005 0.0001",
                },
            )
            self.assets.add(
                Asset(
                    "mesh",
                    {
                        "name": f"{name}_part_{i}",
                        "file": str(part.resolve()),
                    },
                )
            )

        material = Asset("material", {})
        if self.texture:
            material.set("texture", f"{self.texture}_texture")
            material.set("name", f"{self.texture}_material")
            material.set("texuniform", "true")
            material.set("texrepeat", " 5 5")
            material.set("shininess", "1.0")
        else:
            material.set("name", "body_material")
        self.assets.add(material)

        if not fixed:
            ET.SubElement(
                self,
                "joint",
                {"name": f"{name}_{self.COUNTER}_freejoint", "type": "free"},
            )
        StlBody.COUNTER += 1

    def _add_texture(self, texture_type: str):
        texture_path = TEXTURE_FOLDER / texture_type
        if not texture_path.exists():
            raise FileNotFoundError(f"Texture '{texture_type}' does not exist.")
        # select random texture from folder
        texture_files = list(texture_path.glob("*.png"))
        if len(texture_files) == 0:
            raise FileNotFoundError(
                f"Texture folder '{texture_type}' has no texture files."
            )
        selected_texture: pathlib.Path = np.random.choice(texture_files)
        texture_filename = selected_texture.stem

        self.assets.add(
            Asset(
                "texture",
                {
                    "name": f"{texture_filename}_texture",
                    "file": str(selected_texture.resolve()),
                    "type": "2d",
                },
            )
        )
        self.texture = texture_filename

    def get_trimesh(self) -> trimesh.Trimesh:
        decomposed_path = (
            BIN_PICKING_CACHE_FOLDER / self.attrib["name"].rsplit("_", 1)[0]
        )
        parts = [
            trimesh.load_mesh(part)
            for part in decomposed_path.glob(
                f"{self.attrib['name'].rsplit('_', 1)[0]}_part_*.stl"
            )
        ]
        combined = trimesh.util.concatenate(parts)
        return combined

    def get_dimensions(self) -> tuple[float, float, float]:
        mesh = self.get_trimesh()
        bbox = mesh.bounding_box.extents
        return float(bbox[0]), float(bbox[1]), float(bbox[2])


class StlBatch:
    def __init__(
        self,
        stl_body_name: str,
        count: int,
        center: tuple[float, float, float],
        length: float,
        width: float,
        stl_body_attrib={},
        stl_body_texture: str | None = None,
        stl_body_fixed=False,
        spacing: float = 0.02,
    ):
        self.stl_body_name = stl_body_name
        self.count = count
        self.center = center
        self.length = length
        self.width = width
        self.stl_body_attrib = stl_body_attrib
        self.stl_body_texture = stl_body_texture
        self.stl_body_fixed = stl_body_fixed
        self.spacing = spacing

    def next_body(self) -> tuple[trimesh.Trimesh, StlBody]:
        body = StlBody(
            self.stl_body_name,
            fixed=self.stl_body_fixed,
            texture_type=self.stl_body_texture,
            **self.stl_body_attrib,
        )
        rotation = np.random.rand(3) * 2 * np.pi
        mesh = body.get_trimesh().apply_transform(
            trimesh.transformations.euler_matrix(*rotation)
        )
        body.rotate(*[angle * 360 / 2 / np.pi for angle in rotation])
        return mesh, body

    def generate(self):
        bodies = []
        max_y_extend = 0
        max_z_extend = 0
        actual_count = 0
        z = 0
        while actual_count < self.count:
            y = 0
            x = 0
            while y < self.length and actual_count < self.count:
                x = 0
                while x < self.width and actual_count < self.count:
                    next_body_mesh, next_body = self.next_body()
                    bbox = next_body_mesh.bounding_box.extents
                    body_dimensions = (float(bbox[0]), float(bbox[1]), float(bbox[2]))
                    x_max = x + body_dimensions[0]
                    y_max = y + body_dimensions[1]
                    if x_max > self.width:
                        x += body_dimensions[0] + self.spacing
                        break
                    if y_max > self.length:
                        y += max_y_extend + self.spacing
                        break
                    next_body.at(
                        self.center[0] - (self.width / 2) + x + body_dimensions[0] / 2,
                        self.center[1] - (self.length / 2) + y + body_dimensions[1] / 2,
                        self.center[2] + z + body_dimensions[2] / 2,
                    )
                    actual_count += 1
                    if body_dimensions[1] > max_y_extend:
                        max_y_extend = body_dimensions[1]
                    if body_dimensions[2] > max_z_extend:
                        max_z_extend = body_dimensions[2]
                    x += body_dimensions[0] + self.spacing
                    bodies.append(next_body)
                x = self.center[0] - (self.width / 2)
                y += max_y_extend + self.spacing
            z += max_z_extend + self.spacing
        return bodies


class Table(XmlObject):
    COUNTER = 0

    def __init__(
        self,
        length: float = 1.0,
        width: float = 1.0,
        height: float = 0.75,
        fixed: bool = True,
    ):
        super().__init__("body", {"name": f"table_{self.COUNTER}"})
        Table.COUNTER += 1
        ET.SubElement(
            self,
            "geom",
            {
                "name": f"table_top_geom_{self.COUNTER}",
                "type": "box",
                "size": f"{length / 2} {width / 2} {0.025}",
                "pos": f"0 0 {height - 0.025}",
            },
        )
        for i in range(4):
            x = (i % 2 - 0.5) * (length - 0.1)
            y = (i // 2 - 0.5) * (width - 0.1)
            ET.SubElement(
                self,
                "geom",
                {
                    "name": f"table_leg_geom_{self.COUNTER}_{i}",
                    "type": "cylinder",
                    "size": "0.05",
                    "fromto": f"{x} {y} {height - 0.05} {x} {y} 0",
                },
            )
        if not fixed:
            ET.SubElement(
                self,
                "joint",
                {"name": f"table_{self.COUNTER}_freejoint", "type": "free"},
            )


class Box(XmlObject):
    COUNTER = 0

    def __init__(
        self,
        length: float = 0.4,
        width: float = 0.6,
        height: float = 0.4,
        wall_thickness: float = 0.02,
        fixed: bool = False,
    ):
        super().__init__("body", {"name": f"box_{self.COUNTER}"})
        Box.COUNTER += 1
        self.trimesh_primitives = []
        # Bottom
        self.make_box(
            length, width, wall_thickness, 0, 0, wall_thickness / 2, suffix="bottom"
        )
        # Walls
        for i in range(4):
            x = (
                wall_thickness / 2 if i % 2 == 1 else (wall_thickness - length) / 2
            ) * (1 if i < 2 else -1)
            y = (wall_thickness / 2 if i % 2 == 0 else (width - wall_thickness) / 2) * (
                1 if i < 2 else -1
            )
            size_x = wall_thickness if i % 2 == 0 else length - wall_thickness
            size_y = wall_thickness if i % 2 == 1 else width - wall_thickness
            self.make_box(
                size_x,
                size_y,
                height - wall_thickness,
                x,
                y,
                (height + wall_thickness) / 2,
                suffix=f"wall_{i}",
            )
        if not fixed:
            ET.SubElement(
                self,
                "joint",
                {"name": f"box_{self.COUNTER}_freejoint", "type": "free"},
            )

    def make_box(self, length, width, height, x, y, z, suffix=""):
        self.trimesh_primitives.append((length, width, height, x, y, z))
        ET.SubElement(
            self,
            "geom",
            {
                "name": f"box_geom_{self.COUNTER}_{suffix}",
                "type": "box",
                "size": f"{length / 2} {width / 2} {height / 2}",
                "pos": f"{x} {y} {z}",
            },
        )

    def get_trimesh(self) -> trimesh.Trimesh:
        boxes = []
        for length, width, height, x, y, z in self.trimesh_primitives:
            box = trimesh.primitives.Box(extents=(length, width, height))
            box.apply_translation((x, y, z))
            boxes.append(box)
        combined = trimesh.util.concatenate(boxes)
        return combined


class MojocoEnv(XmlObject):
    def __init__(self, name: str = "default_model", objects: list[XmlObject] = []):
        super().__init__("mujoco", {"model": name})
        self.m = None
        self.d = None
        self.worldbody = XmlObject("worldbody")
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

    def get_mujoco(self):
        if self.m is not None and self.d is not None:
            return self.m, self.d
        xml_string = self.xml_spec()
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        self.m = model
        self.d = data
        return model, data

    def run_with_passive_viewer(self, duration_seconds: float = 30.0):
        m, d = self.get_mujoco()
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < duration_seconds:
                step_start = time.time()

                mujoco.mj_step(m, d)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        d.time % 2
                    )

                    viewer.sync()

                elapsed = time.time() - step_start
                if m.opt.timestep > elapsed:
                    time.sleep(m.opt.timestep - elapsed)

    def run_unrendered(self, duration_seconds: float = 30.0):
        m, d = self.get_mujoco()
        start = d.time
        while d.time < start + duration_seconds:
            mujoco.mj_step(m, d)
