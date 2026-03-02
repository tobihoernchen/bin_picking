from xml.etree import ElementTree as ET
import trimesh


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
        if self.tag not in ["body", "geom", "site", "camera", "light"]:
            raise ValueError(f"Cannot set position for tag '{self.tag}'")
        self.set("pos", f"{x} {y} {z}")
        return self

    def rotate(self, x, y, z):
        if self.tag not in ["body", "geom", "site", "camera", "light"]:
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


class XmlObjectCollection:
    def __init__(self, xml_objects: list[XmlObject] = []):
        self.xml_objects = xml_objects


class Asset(XmlObject):
    def __init__(self, tag, attrib=..., **extra):
        super().__init__(tag, attrib, **extra)

    def __hash__(self):
        xml_str = ET.tostring(self, encoding="unicode")
        normalized = xml_str.replace("\n", "").replace(" ", "")
        return hash(normalized)

    def __eq__(self, value):
        return self.__hash__() == value.__hash__()


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
        self.make_box(length, width, wall_thickness, 0, 0, wall_thickness / 2, suffix="bottom")
        # Walls
        for i in range(4):
            x = (wall_thickness / 2 if i % 2 == 1 else (wall_thickness - length) / 2) * (
                1 if i < 2 else -1
            )
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
