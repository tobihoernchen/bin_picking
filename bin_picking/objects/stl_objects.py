from xml.etree import ElementTree as ET
import pathlib
import numpy as np
import trimesh

from bin_picking.objects.objects import Asset, XmlObject


BIN_PICKING_CACHE_FOLDER = pathlib.Path(__file__).parent / ".bin_picking/"
TEXTURE_FOLDER = pathlib.Path(__file__).parent / "textures"


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

    def __init__(self, name, attrib={}, fixed=False, texture_type: str | None = None, **extra):
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
            ET.SubElement(
                self,
                "geom",
                {
                    "type": "mesh",
                    "mesh": f"{name}_part_{i}",
                    "material": "body_material" if not self.texture else f"{self.texture}_material",
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
            raise FileNotFoundError(f"Texture folder '{texture_type}' has no texture files.")
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
        decomposed_path = BIN_PICKING_CACHE_FOLDER / self.attrib["name"].rsplit("_", 1)[0]
        parts = [
            trimesh.load_mesh(part)
            for part in decomposed_path.glob(f"{self.attrib['name'].rsplit('_', 1)[0]}_part_*.stl")
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
        mesh = body.get_trimesh().apply_transform(trimesh.transformations.euler_matrix(*rotation))
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
