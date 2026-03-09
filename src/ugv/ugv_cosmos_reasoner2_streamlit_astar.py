#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import heapq
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import streamlit as st
from PIL import Image, ImageDraw

try:
    import requests
except Exception:
    requests = None

try:
    from owlready2 import get_ontology
except Exception:
    get_ontology = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler((0.0, 0.0, yaw))


def distance2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str, base_dir: Path) -> str:
    pth = Path(path_str).expanduser()
    if not pth.is_absolute():
        pth = (base_dir / pth).resolve()
    return str(pth)


@dataclass
class ProjectPaths:
    base_dir: Path
    ontology_dir: Path
    data_dir: Path
    logs_dir: Path
    ycb_dir: Path
    models_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Optional[str] = None) -> "ProjectPaths":
        root = Path(base_dir).expanduser().resolve() if base_dir else Path(__file__).resolve().parent
        data_dir = root / "data"
        return cls(
            base_dir=root,
            ontology_dir=root / "Ontology",
            data_dir=data_dir,
            logs_dir=data_dir / "logs",
            ycb_dir=data_dir / "YCB_Dataset",
            models_dir=root / "models",
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "base_dir": str(self.base_dir),
            "ontology_dir": str(self.ontology_dir),
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "ycb_dir": str(self.ycb_dir),
            "models_dir": str(self.models_dir),
        }


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------

class EventLogger:
    def __init__(self) -> None:
        self.events: List[Dict] = []

    def log(self, kind: str, message: str, **payload) -> None:
        self.events.append(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": kind,
                "message": message,
                "payload": payload,
            }
        )

    def as_text(self) -> str:
        lines = []
        for ev in self.events:
            tail = f" | {json.dumps(ev['payload'], ensure_ascii=False)}" if ev["payload"] else ""
            lines.append(f"[{ev['ts']}] [{ev['kind']}] {ev['message']}{tail}")
        return "\n".join(lines)

    def recent_text(self, limit: int = 80) -> str:
        lines = []
        for ev in self.events[-limit:]:
            tail = f" | {json.dumps(ev['payload'], ensure_ascii=False)}" if ev["payload"] else ""
            lines.append(f"[{ev['ts']}] [{ev['kind']}] {ev['message']}{tail}")
        return "\n".join(lines)

    def recent_options(self, limit: int = 80) -> List[str]:
        out = []
        for ev in self.events[-limit:]:
            out.append(f"[{ev['ts']}] [{ev['kind']}] {ev['message']}")
        return out

    def as_json(self) -> str:
        return json.dumps(self.events, indent=2, ensure_ascii=False)

    def save(self, output_dir: str, stem: str = "run_log") -> Tuple[str, str]:
        ensure_dir(output_dir)
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        json_path = os.path.join(output_dir, f"{stem}.json")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.as_text())
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.as_json())
        return txt_path, json_path


# -----------------------------------------------------------------------------
# Ontology (OWL only)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticObject:
    name: str
    category: str
    home_room: str
    color_rgba: Tuple[float, float, float, float]
    ycb_candidates: Tuple[str, ...]


class HouseOntology:
    """
    Fallback ontology used only when no valid OWL file is found.
    The simulation objects are always intended to be YCB objects.
    """

    def __init__(self) -> None:
        self.rooms = ["kitchen", "bathroom", "living_room", "bedroom"]
        self.objects: Dict[str, SemanticObject] = {
            "mug_red": SemanticObject(
                "mug_red",
                "mug",
                "kitchen",
                (0.85, 0.15, 0.15, 1.0),
                ("025_mug", "YcbMug"),
            ),
            "soap_blue": SemanticObject(
                "soap_blue",
                "cleanser",
                "bathroom",
                (0.15, 0.35, 0.95, 1.0),
                ("021_bleach_cleanser", "YcbBleachCleanser"),
            ),
            "remote_black": SemanticObject(
                "remote_black",
                "marker",
                "living_room",
                (0.15, 0.15, 0.15, 1.0),
                ("040_large_marker",),
            ),
            "pillow_green": SemanticObject(
                "pillow_green",
                "foam_brick",
                "bedroom",
                (0.15, 0.80, 0.25, 1.0),
                ("061_foam_brick", "YcbFoamBrick"),
            ),
        }

    def target_room_for(self, object_name: str) -> str:
        return self.objects[object_name].home_room

    def object_names(self) -> List[str]:
        return list(self.objects.keys())


class Owlready2OntologyAdapter:
    CANDIDATE_PROPERTIES = {
        "belongsin",
        "locatedin",
        "inroom",
        "room",
        "home_room",
        "belongsto",
        "shouldbein",
        "goesto",
    }

    def __init__(self, fallback: HouseOntology):
        self.fallback = fallback

    @staticmethod
    def _safe_name(entity) -> str:
        return getattr(entity, "name", str(entity))

    def load(self, path: str) -> HouseOntology:
        if not path or not os.path.exists(path) or get_ontology is None:
            return self.fallback

        onto = get_ontology(path).load()
        loaded = HouseOntology()
        loaded.objects = dict(self.fallback.objects)
        found_any = False

        for indiv in onto.individuals():
            subj_name = self._safe_name(indiv)
            if subj_name not in loaded.objects:
                continue

            for prop in indiv.get_properties():
                prop_name = self._safe_name(prop).lower()
                if prop_name not in self.CANDIDATE_PROPERTIES:
                    continue
                try:
                    values = list(prop[indiv])
                except Exception:
                    values = []
                if not values:
                    continue

                room_name = self._safe_name(values[0])
                prev = loaded.objects[subj_name]
                loaded.objects[subj_name] = SemanticObject(
                    prev.name,
                    prev.category,
                    room_name,
                    prev.color_rgba,
                    prev.ycb_candidates,
                )
                found_any = True
                break

        return loaded if found_any else self.fallback


# -----------------------------------------------------------------------------
# World and YCB loading
# -----------------------------------------------------------------------------

@dataclass
class RoomInfo:
    name: str
    center_xy: Tuple[float, float]
    floor_rgba: Tuple[float, float, float, float]
    pedestal_pos: Tuple[float, float, float]
    pedestal_body_id: int = -1


@dataclass
class SimObjectState:
    name: str
    body_id: int
    current_room: str
    on_pedestal_of: str
    picked: bool = False
    ycb_name: Optional[str] = None
    category: Optional[str] = None
    loaded_from_ycb: bool = False


class YCBObjectDB:
    """
    Local YCB loader.

    Supported structures include examples like:
      <dataset_dir>/025_mug/model.urdf
      <dataset_dir>/YcbMug/model.urdf
      <dataset_dir>/025_mug/textured_simple.obj
      <dataset_dir>/YcbMug/textured_simple.obj
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir.strip()

    def find_model_path(self, candidates: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
        if not self.dataset_dir or not os.path.isdir(self.dataset_dir):
            return None, None

        for cand in candidates:
            base = os.path.join(self.dataset_dir, cand)
            if not os.path.isdir(base):
                continue

            urdf = os.path.join(base, "model.urdf")
            if os.path.exists(urdf):
                return urdf, cand

            for fname in ("textured_simple.obj", "textured.obj", "mesh.obj"):
                obj = os.path.join(base, fname)
                if os.path.exists(obj):
                    return obj, cand

        return None, None


class HouseWorld:
    def __init__(
        self,
        client_id: int,
        ontology: HouseOntology,
        ycb_dataset_dir: str,
        logger: Optional[EventLogger] = None,
    ):
        self.cid = client_id
        self.ontology = ontology
        self.room_size = 2.8
        self.rooms: Dict[str, RoomInfo] = {}
        self.objects: Dict[str, SimObjectState] = {}
        self.static_boxes: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.ycb_db = YCBObjectDB(ycb_dataset_dir)
        self.logger = logger
        self._build_world()

    def _log(self, kind: str, message: str, **payload) -> None:
        if self.logger is not None:
            self.logger.log(kind, message, **payload)

    def _create_box(self, half_extents, mass, pos, rgba, collision=True):
        col = (
            p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.cid)
            if collision
            else -1
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.cid
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=self.cid,
        )

    def _register_aabb(self, center, half_extents, inflate=0.0):
        cx, cy, _ = center
        hx, hy, _ = half_extents
        self.static_boxes.append(
            ((cx - hx - inflate, cy - hy - inflate), (cx + hx + inflate, cy + hy + inflate))
        )

    def _create_ycb_or_fallback(self, semantic_obj: SemanticObject, pos) -> Tuple[int, Optional[str], bool]:
        model_path, ycb_name = self.ycb_db.find_model_path(semantic_obj.ycb_candidates)

        if model_path and model_path.endswith(".urdf"):
            body_id = p.loadURDF(
                model_path,
                basePosition=pos,
                globalScaling=0.9,
                useFixedBase=False,
                physicsClientId=self.cid,
            )
            self._log(
                "YCB",
                "Loaded YCB URDF",
                semantic_name=semantic_obj.name,
                ycb_name=ycb_name,
                model_path=model_path,
            )
            return body_id, ycb_name, True

        if model_path and model_path.endswith(".obj"):
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=model_path,
                meshScale=[0.8, 0.8, 0.8],
                rgbaColor=semantic_obj.color_rgba,
                physicsClientId=self.cid,
            )
            col = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=model_path,
                meshScale=[0.8, 0.8, 0.8],
                physicsClientId=self.cid,
            )
            body_id = p.createMultiBody(
                baseMass=0.12,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.cid,
            )
            self._log(
                "YCB",
                "Loaded YCB mesh",
                semantic_name=semantic_obj.name,
                ycb_name=ycb_name,
                model_path=model_path,
            )
            return body_id, ycb_name, True

        body_id = self._create_box([0.06, 0.06, 0.06], 0.15, pos, semantic_obj.color_rgba, collision=True)
        self._log(
            "WARN",
            "YCB model not found; fallback primitive used",
            semantic_name=semantic_obj.name,
            expected_candidates=semantic_obj.ycb_candidates,
        )
        return body_id, None, False

    def _build_world(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=150,
            numSubSteps=4,
            physicsClientId=self.cid,
        )
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        room_defs = [
            ("kitchen", (-2.0, 2.0), (0.95, 0.85, 0.70, 1.0)),
            ("bathroom", (2.0, 2.0), (0.75, 0.88, 0.98, 1.0)),
            ("living_room", (-2.0, -2.0), (0.78, 0.92, 0.78, 1.0)),
            ("bedroom", (2.0, -2.0), (0.93, 0.80, 0.92, 1.0)),
        ]

        for name, center_xy, floor_rgba in room_defs:
            cx, cy = center_xy
            self._create_box(
                [self.room_size / 2, self.room_size / 2, 0.01],
                0,
                [cx, cy, 0.0],
                floor_rgba,
                collision=False,
            )
            pedestal_pos = (cx, cy, 0.20)
            pedestal_half = [0.18, 0.18, 0.20]
            pedestal_id = self._create_box(
                pedestal_half, 0, pedestal_pos, (0.65, 0.65, 0.65, 1.0), collision=True
            )
            self._register_aabb(pedestal_pos, pedestal_half, inflate=0.10)
            self.rooms[name] = RoomInfo(name, center_xy, floor_rgba, pedestal_pos, pedestal_id)

        self._build_walls()
        self._spawn_objects()

    def _build_walls(self) -> None:
        wall_rgba = (0.85, 0.85, 0.85, 1.0)
        walls = [
            ([6.0, 0.08, 0.35], [0.0, 4.0, 0.35]),
            ([6.0, 0.08, 0.35], [0.0, -4.0, 0.35]),
            ([0.08, 6.0, 0.35], [4.0, 0.0, 0.35]),
            ([0.08, 6.0, 0.35], [-4.0, 0.0, 0.35]),
            ([0.08, 1.3, 0.35], [0.0, 2.7, 0.35]),
            ([0.08, 1.3, 0.35], [0.0, -2.7, 0.35]),
            ([1.3, 0.08, 0.35], [2.7, 0.0, 0.35]),
            ([1.3, 0.08, 0.35], [-2.7, 0.0, 0.35]),
        ]
        for half_extents, pos in walls:
            self._create_box(half_extents, 0, pos, wall_rgba, collision=True)
            self._register_aabb(pos, half_extents, inflate=0.08)

    def _spawn_objects(self) -> None:
        wrong_pedestal_for = {
            "mug_red": "bathroom",
            "soap_blue": "living_room",
            "remote_black": "bedroom",
            "pillow_green": "kitchen",
        }

        for obj_name, sem_obj in self.ontology.objects.items():
            wrong_room = wrong_pedestal_for.get(obj_name, "bathroom")
            px, py, pz = self.rooms[wrong_room].pedestal_pos
            body_id, ycb_name, loaded_from_ycb = self._create_ycb_or_fallback(sem_obj, [px, py, pz + 0.19])
            self.objects[obj_name] = SimObjectState(
                name=obj_name,
                body_id=body_id,
                current_room=wrong_room,
                on_pedestal_of=wrong_room,
                picked=False,
                ycb_name=ycb_name,
                category=sem_obj.category,
                loaded_from_ycb=loaded_from_ycb,
            )

    def room_from_xy(self, x: float, y: float) -> str:
        if x < 0 and y > 0:
            return "kitchen"
        if x >= 0 and y > 0:
            return "bathroom"
        if x < 0 and y <= 0:
            return "living_room"
        return "bedroom"

    def update_object_rooms(self) -> None:
        for obj in self.objects.values():
            pos, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self.cid)
            obj.current_room = self.room_from_xy(pos[0], pos[1])
            obj.on_pedestal_of = self.nearest_pedestal_room(pos[0], pos[1])

    def nearest_pedestal_room(self, x: float, y: float) -> str:
        best_room = "kitchen"
        best_dist = 1e9
        for room_name, room in self.rooms.items():
            d = distance2d((x, y), room.pedestal_pos[:2])
            if d < best_dist:
                best_dist = d
                best_room = room_name
        return best_room

    def all_objects_sorted(self) -> bool:
        self.update_object_rooms()
        return all(
            self.objects[name].current_room == self.ontology.target_room_for(name)
            for name in self.objects
        )

    def pedestal_drop_pose(self, room_name: str) -> Tuple[float, float, float]:
        px, py, pz = self.rooms[room_name].pedestal_pos
        return px, py, pz + 0.26

    def object_pose(self, object_name: str) -> Tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(
            self.objects[object_name].body_id, physicsClientId=self.cid
        )
        return float(pos[0]), float(pos[1]), float(pos[2])


# -----------------------------------------------------------------------------
# YOLO integration
# -----------------------------------------------------------------------------

class YoloDetector:
    def __init__(self, enabled: bool, model_path: str, conf: float, logger: EventLogger):
        self.enabled = enabled
        self.model_path = model_path.strip()
        self.conf = conf
        self.logger = logger
        self.model = None

        if self.enabled and YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.logger.log("YOLO", "YOLO model loaded", model_path=self.model_path)
            except Exception as exc:
                self.logger.log(
                    "WARN",
                    "YOLO load failed; detector disabled",
                    error=str(exc),
                    model_path=self.model_path,
                )
                self.enabled = False

    def detect(self, rgb_np: np.ndarray) -> Dict:
        if not self.enabled or self.model is None:
            return {"enabled": False, "detections": []}
        try:
            result = self.model.predict(source=rgb_np, conf=self.conf, verbose=False)[0]
            detections = []
            names = result.names if hasattr(result, "names") else {}
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "label": names.get(cls_id, str(cls_id)),
                            "confidence": round(conf, 4),
                            "xyxy": [round(v, 1) for v in xyxy],
                        }
                    )
            self.logger.log("YOLO", "Detections computed", count=len(detections), detections=detections)
            return {"enabled": True, "detections": detections}
        except Exception as exc:
            self.logger.log("WARN", "YOLO inference failed", error=str(exc))
            return {"enabled": True, "detections": []}

    @staticmethod
    def draw(rgb_np: np.ndarray, detections: List[Dict]) -> Image.Image:
        img = Image.fromarray(rgb_np.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(img)
        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=3)
            draw.text((x1 + 3, y1 + 3), f"{det['label']} {det['confidence']:.2f}", fill=(255, 255, 0))
        return img


# -----------------------------------------------------------------------------
# Cosmos Reasoner2 client
# -----------------------------------------------------------------------------

class CosmosReasoner2Client:
    DEFAULT_PROMPT = (
        "You are Cosmos-Reason2 acting as a high-level reasoning module for an indoor UGV. "
        "Use the scene image, robot pose, current mission, YOLO detections, and ontology mapping. "
        "Return ONLY valid JSON with keys: scene_summary, current_room, visible_objects, hazards, "
        "recommended_next_action, best_target_object, best_target_room, justification. "
        "Each item in visible_objects must contain: name, category, distance_m, bearing_deg, estimated_room."
    )

    def __init__(
        self,
        world: HouseWorld,
        use_real_api: bool,
        api_url: str,
        logger: EventLogger,
        prompt_template: str,
    ):
        self.world = world
        self.use_real_api = use_real_api
        self.api_url = api_url.strip()
        self.logger = logger
        self.prompt_template = prompt_template.strip() or self.DEFAULT_PROMPT

    def _encode_image(self, rgb_np: np.ndarray) -> str:
        pil = Image.fromarray(rgb_np.astype(np.uint8))
        return base64.b64encode(pil_to_png_bytes(pil)).decode("utf-8")

    def _mock_reason(
        self,
        robot_xyyaw: Tuple[float, float, float],
        mission: str,
        yolo_result: Optional[Dict],
        ontology_map: Dict[str, str],
    ) -> Dict:
        rx, ry, ryaw = robot_xyyaw
        visible = []
        hazards = []

        for obj_name in self.world.objects:
            ox, oy, _ = self.world.object_pose(obj_name)
            d = distance2d((rx, ry), (ox, oy))
            bearing = math.atan2(oy - ry, ox - rx)
            rel = wrap_angle(bearing - ryaw)
            if d < 2.6 and abs(rel) < math.radians(80):
                visible.append(
                    {
                        "name": obj_name,
                        "category": self.world.ontology.objects[obj_name].category,
                        "distance_m": round(d, 3),
                        "bearing_deg": round(math.degrees(rel), 1),
                        "estimated_room": self.world.objects[obj_name].current_room,
                    }
                )

        if abs(rx) < 0.35 or abs(ry) < 0.35:
            hazards.append("near_doorway_or_crossing")

        current_room = self.world.room_from_xy(rx, ry)
        best_target_object = visible[0]["name"] if visible else None
        best_target_room = ontology_map.get(best_target_object) if best_target_object else None

        result = {
            "scene_summary": f"Robot is in {current_room} and sees {len(visible)} relevant objects.",
            "current_room": current_room,
            "visible_objects": visible,
            "hazards": hazards,
            "recommended_next_action": "inspect_visible_objects" if visible else "continue_navigation",
            "best_target_object": best_target_object,
            "best_target_room": best_target_room,
            "justification": "Mock reasoner used scene geometry and ontology mapping.",
            "mission": mission,
            "yolo": yolo_result or {"enabled": False, "detections": []},
        }
        self.logger.log("REASONER", "Mock Cosmos-Reason2 result generated", result=result)
        return result

    def reason(
        self,
        rgb_np: np.ndarray,
        robot_xyyaw: Tuple[float, float, float],
        mission: str,
        yolo_result: Optional[Dict],
        ontology_map: Dict[str, str],
    ) -> Dict:
        if not self.use_real_api or not self.api_url or requests is None:
            return self._mock_reason(robot_xyyaw, mission, yolo_result, ontology_map)

        prompt = (
            f"{self.prompt_template}\n\n"
            f"Mission: {mission}\n"
            f"Robot state: x={robot_xyyaw[0]:.2f}, y={robot_xyyaw[1]:.2f}, yaw_deg={math.degrees(robot_xyyaw[2]):.1f}\n"
            f"YOLO detections: {json.dumps(yolo_result or {'enabled': False, 'detections': []}, ensure_ascii=False)}\n"
            f"Ontology mapping: {json.dumps(ontology_map, ensure_ascii=False)}"
        )
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._encode_image(rgb_np)}"},
                        },
                    ],
                }
            ],
            "max_tokens": 700,
            "temperature": 0.1,
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=90)
            res.raise_for_status()
            data = res.json()
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            self.logger.log("REASONER", "Real Cosmos-Reason2 API result received", result=parsed)
            return parsed
        except Exception as exc:
            self.logger.log("WARN", "Real Cosmos-Reason2 call failed, fallback to mock", error=str(exc))
            return self._mock_reason(robot_xyyaw, mission, yolo_result, ontology_map)


# -----------------------------------------------------------------------------
# A* planner
# -----------------------------------------------------------------------------

@dataclass(order=True)
class PriorityNode:
    priority: float
    node: Tuple[int, int] = field(compare=False)


class GridAStar:
    def __init__(self, world: HouseWorld, bounds=(-4.2, 4.2, -4.2, 4.2), resolution=0.14, robot_radius=0.24):
        self.world = world
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.w = int(round((self.xmax - self.xmin) / self.resolution)) + 1
        self.h = int(round((self.ymax - self.ymin) / self.resolution)) + 1
        self.occ = np.zeros((self.h, self.w), dtype=np.uint8)
        self._build_occupancy()

    def _grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        ix, iy = cell
        return self.xmin + ix * self.resolution, self.ymin + iy * self.resolution

    def world_to_grid(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        ix = int(round((x - self.xmin) / self.resolution))
        iy = int(round((y - self.ymin) / self.resolution))
        return int(clamp(ix, 0, self.w - 1)), int(clamp(iy, 0, self.h - 1))

    def _mark_box(self, aabb_min: Tuple[float, float], aabb_max: Tuple[float, float]) -> None:
        pad = self.robot_radius
        x0, y0 = aabb_min[0] - pad, aabb_min[1] - pad
        x1, y1 = aabb_max[0] + pad, aabb_max[1] + pad
        g0 = self.world_to_grid((x0, y0))
        g1 = self.world_to_grid((x1, y1))
        xmin, xmax = sorted([g0[0], g1[0]])
        ymin, ymax = sorted([g0[1], g1[1]])
        self.occ[ymin:ymax + 1, xmin:xmax + 1] = 1

    def _build_occupancy(self) -> None:
        for aabb_min, aabb_max in self.world.static_boxes:
            self._mark_box(aabb_min, aabb_max)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        out = []
        for dx, dy in moves:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and self.occ[ny, nx] == 0:
                out.append((nx, ny))
        return out

    def nearest_free(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        if self.occ[cell[1], cell[0]] == 0:
            return cell
        for radius in range(1, 20):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cell[0] + dx, cell[1] + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.occ[ny, nx] == 0:
                        return (nx, ny)
        return cell

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        start = self.nearest_free(self.world_to_grid(start_xy))
        goal = self.nearest_free(self.world_to_grid(goal_xy))
        frontier: List[PriorityNode] = [PriorityNode(0.0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            current = heapq.heappop(frontier).node
            if current == goal:
                break
            for nxt in self._neighbors(current):
                step_cost = math.hypot(nxt[0] - current[0], nxt[1] - current[1])
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(nxt, goal)
                    heapq.heappush(frontier, PriorityNode(priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return [goal_xy]

        path_cells = []
        cur = goal
        while cur is not None:
            path_cells.append(cur)
            cur = came_from[cur]
        path_cells.reverse()
        path = [self._grid_to_world(c) for c in path_cells]
        return self._simplify(path)

    def _line_free(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        dist = distance2d(a, b)
        n = max(2, int(dist / (self.resolution * 0.5)))
        for i in range(n + 1):
            t = i / n
            x = a[0] * (1 - t) + b[0] * t
            y = a[1] * (1 - t) + b[1] * t
            gx, gy = self.world_to_grid((x, y))
            if self.occ[gy, gx] != 0:
                return False
        return True

    def _simplify(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        anchor = path[0]
        i = 1
        while i < len(path):
            j = i
            while j < len(path) and self._line_free(anchor, path[j]):
                j += 1
            out.append(path[j - 1])
            anchor = path[j - 1]
            i = j
        if out[-1] != path[-1]:
            out.append(path[-1])
        return out

    def occupancy_image(self, path: Optional[List[Tuple[float, float]]] = None) -> Image.Image:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:, :] = [245, 245, 245]
        img[self.occ == 1] = [50, 50, 50]
        if path:
            for x, y in path:
                gx, gy = self.world_to_grid((x, y))
                img[max(0, gy - 1):min(self.h, gy + 2), max(0, gx - 1):min(self.w, gx + 2)] = [220, 40, 40]
        return Image.fromarray(np.flipud(img), mode="RGB")


# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------

class SimpleUGV:
    def __init__(self, client_id: int, start_pos=(0.0, 0.0, 0.12), start_yaw=0.0, sim_substeps: int = 6):
        self.cid = client_id
        self.body_id = self._build_robot(start_pos, start_yaw)
        self.carry_height = 0.30
        self.max_lin = 1.0
        self.max_ang = 2.2
        self.sim_substeps = sim_substeps

    def _build_robot(self, start_pos, start_yaw) -> int:
        chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.18, 0.14, 0.06], physicsClientId=self.cid)
        chassis_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.14, 0.06], rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=self.cid)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.03, rgbaColor=[0.05, 0.05, 0.05, 1.0], physicsClientId=self.cid)
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.03, physicsClientId=self.cid)
        return p.createMultiBody(
            baseMass=4.0,
            baseCollisionShapeIndex=chassis_col,
            baseVisualShapeIndex=chassis_vis,
            basePosition=start_pos,
            baseOrientation=quat_from_yaw(start_yaw),
            linkMasses=[0.1, 0.1],
            linkCollisionShapeIndices=[wheel_col, wheel_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis],
            linkPositions=[[0.0, 0.16, -0.02], [0.0, -0.16, -0.02]],
            linkOrientations=[p.getQuaternionFromEuler((math.pi / 2.0, 0.0, 0.0))] * 2,
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]],
            physicsClientId=self.cid,
        )

    def pose(self) -> Tuple[float, float, float]:
        pos, quat = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.cid)
        yaw = p.getEulerFromQuaternion(quat)[2]
        return float(pos[0]), float(pos[1]), float(yaw)

    def _set_velocity(self, lin: float, ang: float) -> None:
        _, _, yaw = self.pose()
        vx = lin * math.cos(yaw)
        vy = lin * math.sin(yaw)
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[vx, vy, 0.0],
            angularVelocity=[0.0, 0.0, ang],
            physicsClientId=self.cid,
        )

    def stop(self) -> None:
        self._set_velocity(0.0, 0.0)

    def step_to_waypoint(self, target_xy: Tuple[float, float], carried_cb=None) -> Tuple[float, float]:
        x, y, yaw = self.pose()
        dx = target_xy[0] - x
        dy = target_xy[1] - y
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        yaw_err = wrap_angle(desired - yaw)

        lin = clamp(0.9 * dist, -self.max_lin, self.max_lin)
        ang = clamp(2.2 * yaw_err, -self.max_ang, self.max_ang)
        if abs(yaw_err) > 0.7:
            lin = 0.0

        self._set_velocity(lin, ang)
        for _ in range(self.sim_substeps):
            p.stepSimulation(physicsClientId=self.cid)
            if carried_cb is not None:
                carried_cb()

        return dist, yaw_err

    def hold_object(self, object_body_id: int) -> None:
        x, y, yaw = self.pose()
        hold_pos = [x + 0.23 * math.cos(yaw), y + 0.23 * math.sin(yaw), self.carry_height]
        p.resetBasePositionAndOrientation(object_body_id, hold_pos, quat_from_yaw(yaw), physicsClientId=self.cid)
        p.resetBaseVelocity(object_body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)


# -----------------------------------------------------------------------------
# Planner / executor
# -----------------------------------------------------------------------------

class OntologyPlannerExecutor:
    def __init__(
        self,
        world: HouseWorld,
        ontology: HouseOntology,
        ugv: SimpleUGV,
        reasoner: CosmosReasoner2Client,
        yolo: YoloDetector,
        astar: GridAStar,
        logger: EventLogger,
    ):
        self.world = world
        self.ontology = ontology
        self.ugv = ugv
        self.reasoner = reasoner
        self.yolo = yolo
        self.astar = astar
        self.logger = logger
        self.current_carried: Optional[str] = None
        self.current_task: Optional[Dict] = None
        self.current_path: List[Tuple[float, float]] = []
        self.current_phase = "idle"
        self.current_waypoint_idx = 0
        self.last_reasoner_output: Dict = {}
        self.last_yolo_output: Dict = {"enabled": False, "detections": []}
        self.last_robot_rgb: Optional[np.ndarray] = None
        self.step_counter = 0
        self.finished = False

    def _select_next_task(self) -> Optional[Dict]:
        self.world.update_object_rooms()
        for obj_name in self.ontology.object_names():
            obj_state = self.world.objects[obj_name]
            target_room = self.ontology.target_room_for(obj_name)
            if obj_state.current_room != target_room:
                return {
                    "task": "reorganize_object",
                    "object": obj_name,
                    "source_room": obj_state.current_room,
                    "target_room": target_room,
                }
        return None

    def _render_robot_camera(self, width=320, height=240) -> np.ndarray:
        x, y, yaw = self.ugv.pose()
        eye = [x, y, 0.70]
        target = [x + math.cos(yaw), y + math.sin(yaw), 0.45]
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=70, aspect=width / height, nearVal=0.05, farVal=10.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            view,
            proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if p.getConnectionInfo(self.world.cid)["connectionMethod"] == p.GUI else p.ER_TINY_RENDERER,
            physicsClientId=self.world.cid,
        )
        rgba = np.reshape(rgba, (height, width, 4))
        return rgba[:, :, :3].astype(np.uint8)

    def _plan_path(self, goal_xy: Tuple[float, float], label: str) -> None:
        x, y, _ = self.ugv.pose()
        self.current_path = self.astar.plan((x, y), goal_xy)
        self.current_waypoint_idx = 0
        self.logger.log("PLAN", f"A* path computed for {label}", goal=goal_xy, path=self.current_path)

    def _advance_navigation(self) -> bool:
        if not self.current_path:
            return True
        if self.current_waypoint_idx >= len(self.current_path):
            self.ugv.stop()
            return True

        target = self.current_path[self.current_waypoint_idx]

        def carried_cb():
            if self.current_carried is not None:
                self.ugv.hold_object(self.world.objects[self.current_carried].body_id)

        dist, yaw_err = self.ugv.step_to_waypoint(target, carried_cb=carried_cb)

        if self.step_counter % 10 == 0:
            x, y, yaw = self.ugv.pose()
            self.logger.log(
                "NAV",
                "Following A* path",
                waypoint_index=self.current_waypoint_idx,
                target=target,
                pose=(round(x, 3), round(y, 3), round(math.degrees(yaw), 1)),
                dist=round(dist, 3),
                yaw_err_deg=round(math.degrees(yaw_err), 1),
            )

        if dist < 0.12:
            self.current_waypoint_idx += 1

        return self.current_waypoint_idx >= len(self.current_path)

    def _pick_object(self, object_name: str) -> None:
        self.current_carried = object_name
        self.world.objects[object_name].picked = True
        self.logger.log("PICK", f"Picked {object_name}")

    def _drop_object(self, object_name: str, target_room: str) -> None:
        drop_pos = self.world.pedestal_drop_pose(target_room)
        p.resetBasePositionAndOrientation(
            self.world.objects[object_name].body_id,
            drop_pos,
            [0, 0, 0, 1],
            physicsClientId=self.world.cid,
        )
        p.resetBaseVelocity(
            self.world.objects[object_name].body_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.world.cid,
        )
        self.current_carried = None
        self.world.objects[object_name].picked = False
        self.world.update_object_rooms()
        self.logger.log("DROP", f"Dropped {object_name} in {target_room}")

    def tick(self) -> None:
        if self.finished:
            return

        self.step_counter += 1

        if self.current_phase == "idle":
            self.current_task = self._select_next_task()
            if self.current_task is None:
                self.finished = True
                self.ugv.stop()
                self.logger.log("DONE", "All objects are correctly sorted")
                return

            obj_name = self.current_task["object"]
            src_room = self.current_task["source_room"]
            goal_xy = self.world.rooms[src_room].pedestal_pos[:2]
            self.logger.log(
                "TASK",
                f"Move {obj_name} from {src_room} to {self.current_task['target_room']}",
            )
            self._plan_path(goal_xy, f"source pedestal of {obj_name}")
            self.current_phase = "nav_to_source"
            return

        if self.current_phase == "nav_to_source":
            if self._advance_navigation():
                self.ugv.stop()
                self.current_phase = "reason_at_source"
            return

        if self.current_phase == "reason_at_source":
            self.last_robot_rgb = self._render_robot_camera()
            self.last_yolo_output = self.yolo.detect(self.last_robot_rgb)
            obj_name = self.current_task["object"]
            mission = f"Find {obj_name}. Determine where it belongs according to the semantic context and ontology."
            self.world.update_object_rooms()
            ontology_map = {name: self.ontology.target_room_for(name) for name in self.ontology.object_names()}
            self.last_reasoner_output = self.reasoner.reason(
                self.last_robot_rgb,
                self.ugv.pose(),
                mission,
                self.last_yolo_output,
                ontology_map,
            )
            inferred_room = self.last_reasoner_output.get("best_target_room") or self.current_task["target_room"]
            self.logger.log(
                "PLAN",
                "Target selected after reasoner call",
                object=obj_name,
                ontology_target=self.current_task["target_room"],
                reasoner_target=inferred_room,
                reasoner=self.last_reasoner_output,
            )
            obj_pos = self.world.object_pose(obj_name)
            self._plan_path((obj_pos[0], obj_pos[1]), f"approach {obj_name}")
            self.current_phase = "approach_object"
            return

        if self.current_phase == "approach_object":
            if self._advance_navigation():
                self.ugv.stop()
                self._pick_object(self.current_task["object"])
                dst_room = self.last_reasoner_output.get("best_target_room") or self.current_task["target_room"]
                self.current_task["resolved_target_room"] = dst_room
                dst_xy = self.world.rooms[dst_room].pedestal_pos[:2]
                self._plan_path(dst_xy, f"carry {self.current_task['object']} to {dst_room}")
                self.current_phase = "carry_to_target"
            return

        if self.current_phase == "carry_to_target":
            if self._advance_navigation():
                self.ugv.stop()
                dst_room = self.current_task.get("resolved_target_room") or self.current_task["target_room"]
                self._drop_object(self.current_task["object"], dst_room)
                self.current_task = None
                self.current_phase = "idle"
            return

    def run_steps(self, num_steps: int) -> None:
        for _ in range(num_steps):
            self.tick()
            if self.finished:
                break


# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

class DemoSystem:
    def __init__(
        self,
        base_dir: str,
        ontology_path: str,
        use_real_api: bool,
        api_url: str,
        prompt_template: str,
        use_yolo: bool,
        yolo_model_path: str,
        yolo_conf: float,
        ycb_dataset_dir: str,
        output_dir: str,
        seed: int = 7,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.paths = ProjectPaths.from_base_dir(base_dir)
        self.base_dir = self.paths.base_dir
        self.ontology_path = resolve_path(ontology_path, self.base_dir)
        self.output_dir = ensure_dir(resolve_path(output_dir, self.base_dir) if output_dir else str(self.paths.logs_dir))
        self.ycb_dataset_dir = resolve_path(ycb_dataset_dir, self.base_dir) if ycb_dataset_dir else str(self.paths.ycb_dir)

        if yolo_model_path.endswith(".pt") and ("/" in yolo_model_path or "\\" in yolo_model_path or yolo_model_path.startswith(".")):
            self.yolo_model_path = resolve_path(yolo_model_path, self.base_dir)
        else:
            self.yolo_model_path = yolo_model_path

        self.cid = p.connect(p.DIRECT)
        self.logger = EventLogger()
        self.logger.log("INIT", "PyBullet DIRECT session started", client_id=self.cid)

        base_ontology = HouseOntology()
        if self.ontology_path and os.path.exists(self.ontology_path) and self.ontology_path.lower().endswith(".owl"):
            ontology = Owlready2OntologyAdapter(base_ontology).load(self.ontology_path)
            self.logger.log("ONTOLOGY", "Ontology loaded from OWL file with owlready2", ontology_path=self.ontology_path)
        else:
            ontology = base_ontology
            self.logger.log(
                "ONTOLOGY",
                "Using built-in fallback ontology",
                ontology_path=self.ontology_path,
                note="Expected an existing .owl file",
            )
        self.ontology = ontology

        self.world = HouseWorld(
            self.cid,
            self.ontology,
            ycb_dataset_dir=self.ycb_dataset_dir,
            logger=self.logger,
        )
        self.ugv = SimpleUGV(self.cid, start_pos=(0.0, 0.0, 0.12), start_yaw=0.0, sim_substeps=6)
        self.astar = GridAStar(self.world)
        self.yolo = YoloDetector(use_yolo, self.yolo_model_path, yolo_conf, self.logger)
        self.reasoner = CosmosReasoner2Client(self.world, use_real_api, api_url, self.logger, prompt_template)
        self.executor = OntologyPlannerExecutor(
            self.world,
            self.ontology,
            self.ugv,
            self.reasoner,
            self.yolo,
            self.astar,
            self.logger,
        )

        self.logger.log(
            "PATHS",
            "Resolved project paths",
            **self.paths.as_dict(),
            ontology_path=self.ontology_path,
            ycb_dataset_dir=self.ycb_dataset_dir,
            output_dir=self.output_dir,
            yolo_model_path=self.yolo_model_path,
        )
        self.logger.save(self.output_dir, "startup_log")

    def close(self) -> None:
        try:
            if p.isConnected(self.cid):
                p.disconnect(self.cid)
        except Exception:
            pass

    def save_logs(self) -> Tuple[str, str]:
        return self.logger.save(self.output_dir, "run_log")

    def render_overview(self, width=900, height=700) -> Image.Image:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.0],
            distance=9.0,
            yaw=42,
            pitch=-60,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.05,
            farVal=20.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )
        rgba = np.reshape(rgba, (height, width, 4)).astype(np.uint8)
        return Image.fromarray(rgba[:, :, :3], mode="RGB")

    def render_robot_cam(self, width=420, height=300) -> Image.Image:
        rgb = self.executor._render_robot_camera(width, height)
        return Image.fromarray(rgb, mode="RGB")

    def render_robot_cam_with_yolo(self, width=420, height=300) -> Image.Image:
        rgb = self.executor._render_robot_camera(width, height)
        yolo_out = self.yolo.detect(rgb)
        self.executor.last_robot_rgb = rgb
        self.executor.last_yolo_output = yolo_out
        return YoloDetector.draw(rgb, yolo_out.get("detections", []))

    def summary(self) -> Dict:
        self.world.update_object_rooms()
        x, y, yaw = self.ugv.pose()
        return {
            "robot_pose": {"x": round(x, 3), "y": round(y, 3), "yaw_deg": round(math.degrees(yaw), 2)},
            "phase": self.executor.current_phase,
            "current_task": self.executor.current_task,
            "carried": self.executor.current_carried,
            "path": self.executor.current_path,
            "path_waypoint_index": self.executor.current_waypoint_idx,
            "reasoner_output": self.executor.last_reasoner_output,
            "yolo_output": self.executor.last_yolo_output,
            "sorted": self.world.all_objects_sorted(),
            "objects": {
                name: {
                    "current_room": st_.current_room,
                    "target_room": self.ontology.target_room_for(name),
                    "picked": st_.picked,
                    "ycb_name": st_.ycb_name,
                    "category": st_.category,
                    "loaded_from_ycb": st_.loaded_from_ycb,
                }
                for name, st_ in self.world.objects.items()
            },
            "output_dir": self.output_dir,
            "paths": self.paths.as_dict(),
            "ontology_path": self.ontology_path,
            "ycb_dataset_dir": self.ycb_dataset_dir,
            "yolo_model_path": self.yolo_model_path,
        }


def get_demo(
    base_dir: str,
    ontology_path: str,
    use_real_api: bool,
    api_url: str,
    prompt_template: str,
    use_yolo: bool,
    yolo_model_path: str,
    yolo_conf: float,
    ycb_dataset_dir: str,
    output_dir: str,
    force_reset: bool = False,
) -> DemoSystem:
    cfg = {
        "base_dir": base_dir,
        "ontology_path": ontology_path,
        "use_real_api": use_real_api,
        "api_url": api_url,
        "prompt_template": prompt_template,
        "use_yolo": use_yolo,
        "yolo_model_path": yolo_model_path,
        "yolo_conf": yolo_conf,
        "ycb_dataset_dir": ycb_dataset_dir,
        "output_dir": output_dir,
    }
    if force_reset or "demo" not in st.session_state or st.session_state.get("demo_cfg") != cfg:
        if "demo" in st.session_state:
            st.session_state.demo.close()
        st.session_state.demo = DemoSystem(**cfg)
        st.session_state.demo_cfg = cfg
    return st.session_state.demo


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="UGV + Cosmos-Reason2 + A* + OWL + YOLO + YCB", layout="wide")
    st.title("UGV + Cosmos-Reason2 + A* + OWL + YCB Reorganization Demo")

    default_paths = ProjectPaths.from_base_dir()

    if "continuous_run" not in st.session_state:
        st.session_state.continuous_run = False

    with st.sidebar:
        st.header("Configuration")
        base_dir = st.text_input("Project base directory", value=str(default_paths.base_dir))
        ontology_path = st.text_input("Ontology OWL path", value=str(default_paths.ontology_dir / "Home_Extended.owl"))
        output_dir = st.text_input("Output/log directory", value=str(default_paths.logs_dir))

        st.subheader("Cosmos-Reason2")
        use_real_api = st.checkbox("Use real Cosmos-Reason2 API", value=False)
        api_url = st.text_input("Cosmos-Reason2 API URL", value="http://localhost:8000/v1/chat/completions")
        prompt_template = st.text_area("Editable Cosmos prompt", value=CosmosReasoner2Client.DEFAULT_PROMPT, height=220)

        st.subheader("YOLO")
        use_yolo = st.checkbox("Enable YOLO", value=False)
        yolo_model_path = st.text_input("YOLO model path (.pt)", value="yolov8n.pt")
        yolo_conf = st.slider("YOLO confidence", min_value=0.05, max_value=0.95, value=0.25)

        st.subheader("YCB dataset")
        ycb_dataset_dir = st.text_input("YCB dataset directory", value=str(default_paths.ycb_dir))

        st.subheader("Execution")
        step_batch = st.slider("Steps per run", min_value=1, max_value=400, value=80)
        auto_step_batch = st.slider("Continuous mode steps per rerun", min_value=10, max_value=600, value=140)
        refresh_delay = st.slider("Continuous mode delay (s)", min_value=0.01, max_value=1.0, value=0.05)
        live_log_lines = st.slider("Visible log lines", min_value=10, max_value=300, value=80)
        reset = st.button("Reset simulation")

    demo = get_demo(
        base_dir=base_dir,
        ontology_path=ontology_path,
        use_real_api=use_real_api,
        api_url=api_url,
        prompt_template=prompt_template,
        use_yolo=use_yolo,
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
        ycb_dataset_dir=ycb_dataset_dir,
        output_dir=output_dir,
        force_reset=reset,
    )

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        if st.button("Run one planner tick"):
            demo.executor.run_steps(1)
    with c2:
        if st.button(f"Run {step_batch} steps"):
            demo.executor.run_steps(step_batch)
    with c3:
        if st.button("Run until done"):
            for _ in range(10000):
                demo.executor.run_steps(1)
                if demo.executor.finished:
                    break
    with c4:
        if st.button("Start continuous"):
            st.session_state.continuous_run = True
    with c5:
        if st.button("Stop continuous"):
            st.session_state.continuous_run = False

    save_col1, save_col2 = st.columns([1, 1])
    with save_col1:
        if st.button("Save logs to disk"):
            txt_path, json_path = demo.save_logs()
            st.success(f"Saved logs: {txt_path} | {json_path}")
    with save_col2:
        st.write("Continuous mode:", "ON" if st.session_state.continuous_run else "OFF")

    if st.session_state.continuous_run and not demo.executor.finished:
        demo.executor.run_steps(auto_step_batch)
        demo.save_logs()
        time.sleep(refresh_delay)
        st.rerun()

    summary = demo.summary()

    st.subheader("Live simulation log")
    log_col1, log_col2 = st.columns([3, 1])
    with log_col1:
        st.text_area(
            "What is happening now",
            demo.logger.recent_text(live_log_lines),
            height=260,
            key="live_log_textbox",
        )
    with log_col2:
        st.write("Recent events")
        st.multiselect(
            "Latest log entries",
            options=demo.logger.recent_options(min(live_log_lines, 120)),
            default=[],
            key="live_log_list",
        )

    left, right = st.columns([1.15, 1.0])
    with left:
        st.subheader("PyBullet environment")
        st.image(demo.render_overview(), use_container_width=True)
        st.subheader("A* occupancy grid and active path")
        st.image(demo.astar.occupancy_image(summary["path"]), use_container_width=True)

    with right:
        st.subheader("Robot camera")
        if use_yolo:
            st.image(demo.render_robot_cam_with_yolo(), use_container_width=True)
        else:
            st.image(demo.render_robot_cam(), use_container_width=True)
        st.subheader("Execution summary")
        st.json(summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Plan", "Reasoner", "YOLO", "Logs", "Ontology"])
    with tab1:
        st.write("Current phase:", summary["phase"])
        st.write("Current task:")
        st.json(summary["current_task"])
        st.write("Current path:")
        st.json(summary["path"])
        st.write("Object states:")
        st.json(summary["objects"])

    with tab2:
        st.write("Editable prompt:")
        st.code(prompt_template, language="text")
        st.write("Latest reasoner output:")
        st.json(summary["reasoner_output"])
        st.caption("If the real API is disabled or unavailable, the app falls back to a mock reasoner with the same JSON contract.")

    with tab3:
        st.write("Latest YOLO output:")
        st.json(summary["yolo_output"])
        st.caption("YOLO is optional. It only runs if ultralytics is installed and the selected model path is valid.")

    with tab4:
        log_txt = demo.logger.as_text()
        log_json = demo.logger.as_json()
        st.text_area("Execution log", log_txt, height=420)
        st.download_button("Download log TXT", log_txt, file_name="run_log.txt", mime="text/plain")
        st.download_button("Download log JSON", log_json, file_name="run_log.json", mime="application/json")

    with tab5:
        st.write("Resolved paths:")
        st.json(summary["paths"])
        st.write("Resolved ontology OWL path:", summary["ontology_path"])
        st.write("Resolved YCB dataset path:", summary["ycb_dataset_dir"])
        st.write("Resolved YOLO model path:", summary["yolo_model_path"])
        st.write("Loaded ontology mapping:")
        st.json({name: demo.ontology.target_room_for(name) for name in demo.ontology.object_names()})
        st.caption("The ontology is handled only as OWL through owlready2. No Turtle export is used.")

    st.caption(
        "Notes: the ontology is handled only as .owl with owlready2. "
        "The simulation tries to load every object from the local YCB dataset. "
        "If a specific model is missing, the log records it and a primitive fallback is used so the demo does not break. "
        "Continuous mode uses larger simulation batches and smaller rerun delay for smoother motion."
    )


if __name__ == "__main__":
    main()

# requirements:
# streamlit
# pybullet
# numpy
# pillow
# requests
# owlready2
# ultralytics  (optional, for YOLO)