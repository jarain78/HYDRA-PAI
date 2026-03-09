#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ontoai_gui_ycb_cosmos_clips_jso800.py

Adapted version of ontoai_gui_ycb_cosmos_clips.py to work ONLY with the
JSO-800 robot URDF.

Main changes:
- Removed Panda / MyCobot selection.
- Added robust JSO-800 URDF loading with local asset resolution.
- Added JSO-800 joint discovery, home pose, IK motion and simple gripper control.
- Execution is now performed only with JSO-800.
- Added dynamic base pose refresh when the base position/orientation settings change.

Run:
  pip install streamlit pybullet owlready2 clipspy ultralytics transformers torch imageio imageio-ffmpeg
  streamlit run ontoai_gui_ycb_cosmos_clips_jso800.py
"""

from __future__ import annotations

import os
import re
import json
import time
import glob
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pybullet as p
import pybullet_data
import streamlit as st


# ============================================================
# UI language (i18n)
# ============================================================
I18N = {
    "en": {
        "app_title": "OntoAI GUI - JSO-800",
        "headline": "Cosmos (high-level) → CLIPS (plan) → PyBullet JSO-800 (execution)",
        "subtitle": "Cosmos uses the PyBullet camera + ontology to decide intents; CLIPS plans; the JSO-800 robot executes.",
        "controls": "Controls",
        "language": "Language",
        "mode": "Mode",
        "mode_single": "Single snapshot",
        "mode_cont": "Continuous",
        "run": "RUN",
        "step": "Step (snapshot)",
        "reset": "Initialize / Reset",
        "execute": "Execute (move robot)",
        "use_cosmos": "Use Cosmos (high-level)",
        "cosmos": "Cosmos",
        "cosmos_model": "Model",
        "cosmos_max_new_tokens": "max_new_tokens",
        "cosmos_task": "Instruction",
        "yolo": "YOLO",
        "yolo_weights": "weights",
        "yolo_conf": "Conf",
        "fallback_sim": "Fallback to simulated detections",
        "ontology": "Ontology",
        "owl_path": "OWL path",
        "owl_upload": "Upload .owl",
        "iri_object": "IRI Object",
        "iri_room": "IRI Room",
        "ycb": "YCB",
        "ycb_repo": "YCB repo dir",
        "ycb_objects": "Objects (comma)",
        "ycb_positions": "Positions (x,y,z per line)",
        "map_title": "YOLO→Ontology mapping (JSON)",
        "camera": "PyBullet camera",
        "press_to_run": "Press **Step (snapshot)** or enable **RUN**.",
        "detections": "Detections + Ontology grounding",
        "no_detections": "No detections. Enable fallback or check YOLO/ontology.",
        "logs": "Logs",
        "cosmos_reasoner": "Cosmos-Reasoner (high-level)",
        "raw_output": "Raw output",
        "parsed_json": "**Parsed JSON**",
        "intents": "**Intents**",
        "enable_cosmos_hint": "Enable 'Use Cosmos' to see reasoning.",
        "clips_planner": "CLIPS planner",
        "execution": "Execution",
        "executed": "Action executed",
        "no_move": "No MOVE tasks planned (CLIPS). Check detections/ontology or Cosmos prompt.",
        "enable_execute_hint": "Enable 'Execute (move robot)' to apply the plan.",
        "urdf_path": "JSO-800 URDF path",
        "use_fixed_base": "Use fixed base",
        "auto_robot_base": "Auto place robot base near object",
        "robot_base_pos": "Robot base position (x, y, z)",
        "robot_base_rpy": "Robot base orientation RPY (deg)",
        "base_pose_changed": "Robot base pose/config changed. Reloading world...",
    },
    "es": {
        "app_title": "OntoAI GUI - JSO-800",
        "headline": "Cosmos (alto nivel) → CLIPS (plan) → PyBullet JSO-800 (ejecución)",
        "subtitle": "Cosmos usa la cámara de PyBullet + ontología para decidir intents; CLIPS planifica; el robot JSO-800 ejecuta.",
        "controls": "Controles",
        "language": "Idioma",
        "mode": "Modo",
        "mode_single": "Solo una imagen",
        "mode_cont": "Continuo",
        "run": "RUN",
        "step": "Paso (snapshot)",
        "reset": "Inicializar / Reset",
        "execute": "Ejecutar (mover robot)",
        "use_cosmos": "Usar Cosmos (alto nivel)",
        "cosmos": "Cosmos",
        "cosmos_model": "Modelo",
        "cosmos_max_new_tokens": "max_new_tokens",
        "cosmos_task": "Instrucción",
        "yolo": "YOLO",
        "yolo_weights": "weights",
        "yolo_conf": "Conf",
        "fallback_sim": "Fallback a detección por simulación",
        "ontology": "Ontología",
        "owl_path": "Ruta OWL",
        "owl_upload": "Cargar .owl",
        "iri_object": "IRI Object",
        "iri_room": "IRI Room",
        "ycb": "YCB",
        "ycb_repo": "YCB repo dir",
        "ycb_objects": "Objetos (comma)",
        "ycb_positions": "Posiciones (x,y,z por línea)",
        "map_title": "Mapeo YOLO→Onto (JSON)",
        "camera": "Cámara PyBullet",
        "press_to_run": "Pulsa **Paso (snapshot)** o activa **RUN**.",
        "detections": "Detecciones + Grounding Ontology",
        "no_detections": "Sin detecciones. Activa fallback o revisa YOLO/ontología.",
        "logs": "Logs",
        "cosmos_reasoner": "Cosmos-Reasoner (alto nivel)",
        "raw_output": "Salida cruda",
        "parsed_json": "**JSON parseado**",
        "intents": "**Intents**",
        "enable_cosmos_hint": "Activa 'Usar Cosmos' para ver razonamiento.",
        "clips_planner": "CLIPS Planner",
        "execution": "Ejecución",
        "executed": "Acción ejecutada",
        "no_move": "No hay tareas MOVE planificadas (CLIPS). Revisa detecciones/ontología o prompt de Cosmos.",
        "enable_execute_hint": "Activa 'Ejecutar (mover robot)' para aplicar el plan.",
        "urdf_path": "Ruta URDF JSO-800",
        "use_fixed_base": "Base fija",
        "auto_robot_base": "Posicionar base del robot automáticamente cerca del objeto",
        "robot_base_pos": "Posición base del robot (x, y, z)",
        "robot_base_rpy": "Orientación base del robot RPY (deg)",
        "base_pose_changed": "La configuración de la base del robot cambió. Recargando el mundo...",
    },
}

def tr(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


# ============================================================
# Defaults
# ============================================================

REPO_URL = "https://github.com/elpis-lab/YCB_Dataset.git"
YCB_REPO_DIR_DEFAULT = "YCB_Dataset"
YCB_OBJECTS_DEFAULT = ["d_cups"]
YCB_BASE_POS_DEFAULT = [(0.55, 0.00, 0.15)]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

def rel_or_abs(pth: str, base: Path) -> str:
    pp = Path(pth)
    return str(pp if pp.is_absolute() else (base / pth).resolve())

ONTO_PATH_DEFAULT = rel_or_abs(os.getenv("HYDRA_ONTO", "Ontology/Home.owl"), REPO_ROOT)
IRI_OBJECT_DEFAULT = "*#Object"
IRI_ROOM_DEFAULT = "*#Room"

YOLO_WEIGHTS_DEFAULT = rel_or_abs(os.getenv("HYDRA_YOLO", "models/yolo11n.pt"), REPO_ROOT)
YOLO_CONF_DEFAULT = 0.35

YOLO_TO_ONTO_DEFAULT = {
    "cup": "cup",
    "cups": "cup",
    "mug": "cup",
    "b_cups": "b_cups",
    "d_cups": "d_cups",
}

JSO800_URDF_DEFAULT = str(Path(os.getenv("HYDRA_JSO800_URDF", "src/RobotModels/JSO-800_v3/JSO_800_v3.urdf")).expanduser().resolve())
ROBOT_USE_FIXED_BASE_DEFAULT = True
ROBOT_AUTO_BASE_DEFAULT = True
ROBOT_BASE_POS_DEFAULT = (0.18, -0.30, 0.0)
ROBOT_BASE_RPY_DEG_DEFAULT = (0.0, 0.0, 0.0)

COSMOS_MODEL_DEFAULT = "nvidia/Cosmos-Reason2-2B"

SIM_HZ = 240
DT = 1.0 / SIM_HZ
MAX_JOINT_VEL = 0.9
DEFAULT_FORCE = 120
GRASP_Z_OFFSET = 0.015


# ============================================================
# Rooms / zones
# ============================================================

@dataclass(frozen=True)
class RoomZone:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    drop_xy: Tuple[float, float]

ROOMS = [
    RoomZone(name="kitchen",  xmin=-0.15, xmax=0.25, ymin=-0.25, ymax=0.25, drop_xy=(0.10,  0.18)),
    RoomZone(name="bathroom", xmin=0.25,  xmax=0.65, ymin=-0.25, ymax=0.25, drop_xy=(0.52, -0.18)),
]
ROOM_NAMES = [r.name for r in ROOMS]

def room_from_xy(x: float, y: float) -> Optional[str]:
    for r in ROOMS:
        if r.xmin <= x <= r.xmax and r.ymin <= y <= r.ymax:
            return r.name
    return None

def drop_point(room_name: str) -> Tuple[float, float]:
    for r in ROOMS:
        if r.name == room_name:
            return r.drop_xy
    return ROOMS[0].drop_xy


# ============================================================
# Logging
# ============================================================

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    st.session_state.setdefault("ui_logs", []).append(line)


# ============================================================
# YCB utilities
# ============================================================

@dataclass
class YCBPaths:
    repo_dir: str
    ycb_dir: str

def ensure_repo(repo_dir: str) -> YCBPaths:
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        log(f"Cloning YCB_Dataset -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, repo_dir])
    ycb_dir = os.path.join(repo_dir, "ycb")
    if not os.path.isdir(ycb_dir):
        raise RuntimeError(f"Cannot find 'ycb/' inside: {repo_dir}")
    return YCBPaths(repo_dir=repo_dir, ycb_dir=ycb_dir)

def list_ycb_objects(ycb_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(ycb_dir) if os.path.isdir(os.path.join(ycb_dir, d)) and not d.startswith(".")])

def find_urdf(obj_dir: str) -> Optional[str]:
    urdfs = glob.glob(os.path.join(obj_dir, "**", "*.urdf"), recursive=True)
    if not urdfs:
        return None
    for key in ("model.urdf", "object.urdf", "textured.urdf"):
        for u in urdfs:
            if os.path.basename(u).lower() == key:
                return u
    return urdfs[0]

def find_mesh(obj_dir: str) -> Optional[str]:
    for ext in ("*.obj", "*.stl", "*.dae"):
        meshes = glob.glob(os.path.join(obj_dir, "**", ext), recursive=True)
        if meshes:
            for key in ("textured.obj", "model.obj", "nontextured.obj", "collision.obj"):
                for m in meshes:
                    if os.path.basename(m).lower() == key:
                        return m
            return meshes[0]
    return None

def load_ycb_object(
    obj_dir: str,
    base_pos: Tuple[float, float, float],
    base_orn_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mass: float = 0.30,
) -> int:
    urdf = find_urdf(obj_dir)
    orn = p.getQuaternionFromEuler(base_orn_rpy)

    if urdf:
        return p.loadURDF(
            urdf,
            basePosition=list(base_pos),
            baseOrientation=orn,
            globalScaling=float(mesh_scale[0]),
            useFixedBase=False,
        )

    mesh = find_mesh(obj_dir)
    if not mesh:
        raise RuntimeError(f"No URDF or mesh found in: {obj_dir}")

    col_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh,
        meshScale=list(mesh_scale),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh,
        meshScale=list(mesh_scale),
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=list(base_pos),
        baseOrientation=orn,
    )


# ============================================================
# Ontology
# ============================================================

@dataclass
class OntoIndexEntry:
    instance: object
    classes: List[str]
    class_path: List[str]
    room: Optional[str]
    robot: Optional[str]
    pickable: bool

def _class_name_safe(c) -> str:
    try:
        return str(c.name)
    except Exception:
        return str(c)

def _ancestors_chain(cls) -> List[str]:
    try:
        ancs = list(cls.ancestors())
        names = [_class_name_safe(a) for a in ancs if _class_name_safe(a) not in ("Thing",)]
        return list(dict.fromkeys(sorted(names)))
    except Exception:
        return []

def load_ontology_index(onto_uri: str, iri_object: str, iri_room: str) -> Tuple[object, Dict[str, OntoIndexEntry], Dict[str, List[str]]]:
    from owlready2 import get_ontology

    onto = get_ontology(onto_uri).load()
    ObjectClass = onto.search_one(iri=iri_object)
    RoomClass = onto.search_one(iri=iri_room)
    if ObjectClass is None or RoomClass is None:
        raise ValueError("Missing 'Object' or 'Room' classes. Adjust IRI_OBJECT / IRI_ROOM.")

    hierarchy: Dict[str, List[str]] = {}
    for cls in onto.classes():
        try:
            cname = cls.name
        except Exception:
            continue
        subs = []
        try:
            for s in cls.subclasses():
                try:
                    subs.append(s.name)
                except Exception:
                    pass
        except Exception:
            pass
        if subs:
            hierarchy[cname] = sorted(list(dict.fromkeys(subs)))

    info: Dict[str, OntoIndexEntry] = {}
    for obj in ObjectClass.instances():
        name = obj.name.lower()

        direct_classes = []
        try:
            for c in obj.is_a:
                if hasattr(c, "name"):
                    direct_classes.append(c.name)
        except Exception:
            pass
        direct_classes = sorted(list(dict.fromkeys(direct_classes)))

        class_path = []
        if direct_classes:
            c0 = onto.search_one(iri="*#" + direct_classes[0])
            if c0 is not None:
                class_path = _ancestors_chain(c0)

        room = None
        if hasattr(obj, "locatedIn") and obj.locatedIn:
            try:
                room = obj.locatedIn[0].name.lower()
            except Exception:
                room = str(obj.locatedIn[0]).lower()

        robot = None
        if hasattr(obj, "canBePickedBy") and obj.canBePickedBy:
            try:
                robot = obj.canBePickedBy[0].name
            except Exception:
                robot = str(obj.canBePickedBy[0])

        info[name] = OntoIndexEntry(
            instance=obj,
            classes=direct_classes,
            class_path=class_path,
            room=room,
            robot=robot,
            pickable=bool(robot),
        )

    return onto, info, hierarchy

def ontology_hierarchy_compact(hierarchy: Dict[str, List[str]], max_edges: int = 120) -> List[Dict[str, str]]:
    edges = []
    for parent, subs in sorted(hierarchy.items()):
        for child in subs:
            edges.append({"parent": parent, "child": child})
            if len(edges) >= max_edges:
                return edges
    return edges

def ontology_objects_compact(onto_info: Dict[str, OntoIndexEntry], max_items: int = 80) -> List[Dict[str, Any]]:
    out = []
    for k, e in sorted(onto_info.items(), key=lambda kv: kv[0])[:max_items]:
        out.append({
            "onto": k,
            "classes": e.classes,
            "class_path": e.class_path,
            "destination_room": e.room,
            "pickable": e.pickable,
            "pickable_by": e.robot,
        })
    if len(onto_info) > max_items:
        out.append({"_note": f"... {len(onto_info)-max_items} more objects omitted"})
    return out


# ============================================================
# Camera
# ============================================================

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.02
    far: float = 3.0
    eye: Tuple[float, float, float] = (0.35, -0.60, 0.55)
    target: Tuple[float, float, float] = (0.35, 0.00, 0.10)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

def render_camera(cam: CameraConfig) -> Tuple[np.ndarray, List[float], List[float]]:
    view = p.computeViewMatrix(cam.eye, cam.target, cam.up)
    proj = p.computeProjectionMatrixFOV(cam.fov, cam.width / cam.height, cam.near, cam.far)
    img = p.getCameraImage(
        cam.width, cam.height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.reshape(img[2], (cam.height, cam.width, 4)).astype(np.uint8)
    return rgba[:, :, :3], view, proj

def save_rgb_image(rgb: np.ndarray, out_path: str) -> str:
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_path, rgb)
        return out_path
    except Exception:
        from PIL import Image
        Image.fromarray(rgb).save(out_path)
        return out_path

def world_to_pixel(pos_xyz, view, proj, width, height) -> Optional[Tuple[float, float]]:
    x, y, z = pos_xyz
    vec = np.array([x, y, z, 1.0], dtype=np.float64)
    V = np.array(view, dtype=np.float64).reshape(4, 4).T
    Pm = np.array(proj, dtype=np.float64).reshape(4, 4).T
    clip = Pm @ (V @ vec)
    if clip[3] == 0:
        return None
    ndc = clip[:3] / clip[3]
    if ndc[2] < -1 or ndc[2] > 1:
        return None
    u = (ndc[0] * 0.5 + 0.5) * width
    v = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return (u, v)

def associate_detection_to_body(
    bbox_xyxy: Tuple[int, int, int, int],
    sim_bodies: Dict[str, int],
    view, proj, width, height
) -> Optional[int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    best_id = None
    best_d2 = 1e18
    for _, bid in sim_bodies.items():
        (bx, by, bz), _ = p.getBasePositionAndOrientation(bid)
        uv = world_to_pixel((bx, by, bz), view, proj, width, height)
        if uv is None:
            continue
        du = uv[0] - cx
        dv = uv[1] - cy
        d2 = du * du + dv * dv
        if d2 < best_d2:
            best_d2 = d2
            best_id = bid
    return best_id


# ============================================================
# CLIPS planner
# ============================================================

def build_clips_planner(env) -> None:
    kb = r"""
(deftemplate obj
  (slot id (type SYMBOL))
  (slot onto (type STRING))
  (slot pickable (type SYMBOL))
  (slot cur_room (type SYMBOL))
  (slot dst_room (type SYMBOL))
)

(deftemplate intent
  (slot onto (type STRING))
  (slot action (type SYMBOL))
  (slot to (type SYMBOL))
  (slot priority (type INTEGER))
  (slot rationale (type STRING))
)

(deftemplate task
  (slot action (type SYMBOL))
  (slot id (type SYMBOL))
  (slot to (type SYMBOL))
  (slot reason (type STRING))
  (slot priority (type INTEGER))
)

(deftemplate trace
  (slot msg (type STRING))
)

(defrule cosmos-intent-move
  (declare (salience 100))
  (intent (onto ?o) (action move) (to ?dr) (priority ?p) (rationale ?rat))
  (obj (id ?id) (onto ?o) (pickable yes) (cur_room ?cr) (dst_room ?dst))
  (test (neq ?dr none))
  (test (neq ?cr unknown))
  (test (neq ?cr ?dr))
  =>
  (assert (task (action move) (id ?id) (to ?dr)
                (reason (str-cat "Cosmos intent: " ?rat))
                (priority ?p)))
  (assert (trace (msg (str-cat "COSMOS MOVE " ?id " " ?cr "->" ?dr " (onto=" ?o ")"))))
)

(defrule cosmos-intent-ignore
  (declare (salience 90))
  (intent (onto ?o) (action ignore) (to ?to) (priority ?p) (rationale ?rat))
  (obj (id ?id) (onto ?o))
  =>
  (assert (task (action ignore) (id ?id) (to ?to)
                (reason (str-cat "Cosmos ignore: " ?rat))
                (priority ?p)))
)

(defrule not-pickable-ignore
  (declare (salience 10))
  (not (intent))
  (obj (id ?id) (onto ?o) (pickable no))
  =>
  (assert (task (action ignore) (id ?id) (to none)
                (reason (str-cat "Not pickable according to ontology: " ?o))
                (priority 999)))
)

(defrule missing-destination-ignore
  (declare (salience 9))
  (not (intent))
  (obj (id ?id) (onto ?o) (pickable yes) (dst_room none))
  =>
  (assert (task (action ignore) (id ?id) (to none)
                (reason (str-cat "Pickable but missing locatedIn/destination: " ?o))
                (priority 999)))
)

(defrule out-of-place-move
  (declare (salience 8))
  (not (intent))
  (obj (id ?id) (onto ?o) (pickable yes) (cur_room ?cr) (dst_room ?dr))
  (test (neq ?cr unknown))
  (test (neq ?dr none))
  (test (neq ?cr ?dr))
  =>
  (assert (task (action move) (id ?id) (to ?dr)
                (reason (str-cat "Out of place: " ?cr " -> " ?dr " (onto=" ?o ")"))
                (priority 100)))
)
"""
    with tempfile.TemporaryDirectory() as td:
        clp = Path(td) / "planner.clp"
        clp.write_text(kb, encoding="utf-8")
        env.batch_star(str(clp))

def clear_clips_dynamic(env) -> None:
    to_retract = []
    for f in env.facts():
        if f.template.name in {"obj", "intent", "task", "trace"}:
            to_retract.append(f)
    for f in to_retract:
        f.retract()

def clips_get_tasks(env) -> List[Dict[str, Any]]:
    out = []
    for f in env.facts():
        if f.template.name == "task":
            out.append({
                "action": str(f["action"]),
                "id": str(f["id"]),
                "to": str(f["to"]),
                "reason": str(f["reason"]),
                "priority": int(f["priority"]),
            })
    out.sort(key=lambda d: (d["priority"], d["id"]))
    return out

def clips_get_traces(env) -> List[str]:
    out = []
    for f in env.facts():
        if f.template.name == "trace":
            out.append(str(f["msg"]))
    return out

def body_symbol(body_id: int) -> str:
    return f"obj_{body_id}"

def symbol_to_body(sym: str) -> Optional[int]:
    m = re.match(r"^obj_(\d+)$", sym.strip())
    return int(m.group(1)) if m else None


# ============================================================
# JSO-800 control
# ============================================================

JSO_ARM_JOINT_NAMES = [
    "jso800_joint_1",
    "jso800_joint_2",
    "jso800_joint_3",
    "jso800_joint_4",
    "jso800_joint_5",
]
JSO_GRIPPER_JOINT_NAME = "jso800_joint_6"

def find_joint_indices_by_name(robot_id: int, wanted: List[str]) -> Dict[str, int]:
    name_to_idx = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        jname = info[1].decode("utf-8")
        if jname in wanted:
            name_to_idx[jname] = j
    return name_to_idx

def get_jso_joint_map(robot_id: int) -> Dict[str, int]:
    return find_joint_indices_by_name(robot_id, JSO_ARM_JOINT_NAMES + [JSO_GRIPPER_JOINT_NAME])

def jso_home(robot_id: int, jmap: Dict[str, int]) -> None:
    home = {
        "jso800_joint_1": 0.0,
        "jso800_joint_2": 0.8,
        "jso800_joint_3": 1.0,
        "jso800_joint_4": 1.0,
        "jso800_joint_5": 0.0,
        "jso800_joint_6": 0.35,
    }
    for jn, q in home.items():
        if jn in jmap:
            jid = jmap[jn]
            p.resetJointState(robot_id, jid, q)
            p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=q, force=DEFAULT_FORCE)
    for _ in range(int(0.5 * SIM_HZ)):
        p.stepSimulation()
        time.sleep(DT)

def jso_open_gripper(robot_id: int, jmap: Dict[str, int]) -> None:
    jid = jmap.get(JSO_GRIPPER_JOINT_NAME, None)
    if jid is None:
        return
    p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=0.35, force=40)

def jso_close_gripper(robot_id: int, jmap: Dict[str, int]) -> None:
    jid = jmap.get(JSO_GRIPPER_JOINT_NAME, None)
    if jid is None:
        return
    p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=0.02, force=60)

def jso_move_ee(
    robot_id: int,
    jmap: Dict[str, int],
    ee_link: int,
    target_pos,
    target_orn,
    duration: float,
    max_joint_vel: float = MAX_JOINT_VEL,
    force: float = DEFAULT_FORCE,
) -> None:
    arm_joint_ids = [jmap[jn] for jn in JSO_ARM_JOINT_NAMES if jn in jmap]
    steps = max(1, int(duration * SIM_HZ))
    max_step = max_joint_vel * DT

    q_ik = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        maxNumIterations=200,
        residualThreshold=1e-4,
    )
    q_goal = [q_ik[j] if j < len(q_ik) else p.getJointState(robot_id, j)[0] for j in arm_joint_ids]
    q = [p.getJointState(robot_id, j)[0] for j in arm_joint_ids]

    for _ in range(steps):
        q_next = []
        for qc, qg in zip(q, q_goal):
            dq = qg - qc
            dq = float(np.clip(dq, -max_step, max_step))
            q_next.append(qc + dq)
        q = q_next

        for jid, qj in zip(arm_joint_ids, q):
            try:
                p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=qj, force=force, maxVelocity=max_joint_vel)
            except TypeError:
                p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=qj, force=force)
        p.stepSimulation()
        time.sleep(DT)

def grasp_with_constraint(robot_id: int, ee_link: int, obj_id: int) -> int:
    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos, ee_orn = ee_state[4], ee_state[5]
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)

    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
    child_frame_pos, child_frame_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos, obj_orn)

    return p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFramePosition=child_frame_pos,
        childFrameOrientation=child_frame_orn,
    )

def execute_pick_and_place(robot_id: int, jmap: Dict[str, int], ee_link: int, obj_id: int, dst_room: str) -> List[str]:
    logs = []
    (ox, oy, oz), _ = p.getBasePositionAndOrientation(obj_id)
    dx, dy = drop_point(dst_room)
    down_orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

    logs.append(f"EXEC JSO-800 pick&place body_id={obj_id} -> '{dst_room}' from=({ox:.3f},{oy:.3f},{oz:.3f}) drop=({dx:.3f},{dy:.3f})")

    jso_open_gripper(robot_id, jmap)
    jso_move_ee(robot_id, jmap, ee_link, [ox, oy, 0.26], down_orn, duration=1.4)

    pre_z = max(0.03, 0.08 - GRASP_Z_OFFSET)
    jso_move_ee(robot_id, jmap, ee_link, [ox, oy, pre_z], down_orn, duration=1.1)

    jso_close_gripper(robot_id, jmap)
    for _ in range(int(0.35 * SIM_HZ)):
        p.stepSimulation()
        time.sleep(DT)

    cid = grasp_with_constraint(robot_id, ee_link, obj_id)
    logs.append(f"GRASP constraint id={cid}")

    jso_move_ee(robot_id, jmap, ee_link, [ox, oy, 0.28], down_orn, duration=1.2)
    jso_move_ee(robot_id, jmap, ee_link, [dx, dy, 0.28], down_orn, duration=2.2)

    place_z = max(0.05, 0.10 - GRASP_Z_OFFSET)
    jso_move_ee(robot_id, jmap, ee_link, [dx, dy, place_z], down_orn, duration=1.1)

    try:
        p.removeConstraint(cid)
    except Exception:
        pass

    jso_open_gripper(robot_id, jmap)
    for _ in range(int(0.35 * SIM_HZ)):
        p.stepSimulation()
        time.sleep(DT)

    logs.append("RELEASE")
    jso_move_ee(robot_id, jmap, ee_link, [dx, dy, 0.28], down_orn, duration=1.0)
    logs.append("DONE")
    return logs


# ============================================================
# Cosmos
# ============================================================

def _load_cosmos_model(model_name: str):
    import torch
    import transformers
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor

def build_cosmos_highlevel_prompt(ontology_objects, ontology_hierarchy_edges, perceived, user_task: str) -> str:
    return (
        "You are a high-level robotic task reasoner for a household environment.\n"
        "You MUST ground decisions on the provided OWL ontology knowledge AND the current perception.\n"
        "You MUST output ONLY valid JSON, no markdown.\n\n"
        "Ontology class hierarchy (subClassOf edges):\n"
        f"{json.dumps(ontology_hierarchy_edges, ensure_ascii=False)}\n\n"
        "Ontology object knowledge (each object -> classes, class_path, destination_room, pickable):\n"
        f"{json.dumps(ontology_objects, ensure_ascii=False)}\n\n"
        "Current perception (detected objects, their current room, and body_id):\n"
        f"{json.dumps(perceived, ensure_ascii=False)}\n\n"
        f"User task / instruction:\n{user_task}\n\n"
        "Return ONLY JSON with this exact schema:\n"
        "{\n"
        "  \"high_level_goal\": string,\n"
        "  \"intents\": [\n"
        "    {\"onto\": string, \"action\": \"move\"|\"ignore\", \"to_room\": string|\"none\", \"priority\": int, \"rationale\": string}\n"
        "  ],\n"
        "  \"notes\": string\n"
        "}\n\n"
        "Rules:\n"
        "- Only reference objects present in Current perception.\n"
        "- If an object is not pickable (per ontology), action MUST be ignore.\n"
        "- If destination_room is missing in ontology, action MUST be ignore.\n"
        "- If cur_room equals destination_room, action SHOULD be ignore.\n"
        "- Use lower priority number = more urgent (1 is highest).\n"
    )

def _extract_first_json(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{[\s\S]*\})", text)
    return m.group(1).strip() if m else None

def parse_cosmos_intents(cosmos_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    js = _extract_first_json(cosmos_text)
    if not js:
        return {}, []
    try:
        data = json.loads(js)
    except Exception:
        return {}, []
    intents = data.get("intents", []) if isinstance(data, dict) else []
    out = []
    if isinstance(intents, list):
        for it in intents:
            if not isinstance(it, dict):
                continue
            onto = str(it.get("onto", "")).strip().lower()
            action = str(it.get("action", "ignore")).strip().lower()
            to_room = str(it.get("to_room", "none")).strip().lower()
            pr = it.get("priority", 999)
            try:
                pr = int(pr)
            except Exception:
                pr = 999
            rat = str(it.get("rationale", "")).replace('"', "'").strip()
            if not onto:
                continue
            if action not in {"move", "ignore"}:
                action = "ignore"
            out.append({"onto": onto, "action": action, "to_room": to_room, "priority": pr, "rationale": rat})
    out.sort(key=lambda x: (x["priority"], x["onto"]))
    return (data if isinstance(data, dict) else {}), out

def run_cosmos_highlevel_on_image(image_path: str, prompt_text: str, model_name: str, max_new_tokens: int):
    import torch
    if "cosmos_model" not in st.session_state:
        st.session_state["cosmos_model"] = _load_cosmos_model(model_name)
    model, processor = st.session_state["cosmos_model"]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a high-level robotic reasoner. Output ONLY valid JSON."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]

    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    parsed, intents = parse_cosmos_intents(raw_text)
    return raw_text, parsed, intents


# ============================================================
# Perception fallback
# ============================================================

def synthesize_detections_from_sim(sim_objects: Dict[str, int], onto_info: Dict[str, OntoIndexEntry]) -> List[Dict[str, Any]]:
    onto_keys = set(onto_info.keys())
    dets = []
    only_one = len(sim_objects) == 1
    cup_exists = "cup" in onto_keys

    for sim_name, body_id in sim_objects.items():
        sim_l = str(sim_name).lower()
        onto_key = sim_l if sim_l in onto_keys else None
        if onto_key is None:
            for k in onto_keys:
                if k in sim_l:
                    onto_key = k
                    break
        if onto_key is None and only_one and cup_exists:
            onto_key = "cup"
        if onto_key is None:
            continue

        entry = onto_info.get(onto_key)
        (px, py, _), _ = p.getBasePositionAndOrientation(body_id)
        cur_room = room_from_xy(px, py) or "unknown"
        dets.append({
            "source": "sim",
            "yolo": "sim",
            "onto": onto_key,
            "body_id": int(body_id),
            "cur_room": cur_room,
            "dst_room": (entry.room.lower() if entry and entry.room else "none"),
            "pickable": bool(entry.pickable) if entry else False,
            "conf": 1.0,
            "bbox": None,
        })
    return dets


# ============================================================
# World setup
# ============================================================

def draw_rooms_debug() -> None:
    z = 0.001
    for r in ROOMS:
        corners = [(r.xmin, r.ymin, z), (r.xmax, r.ymin, z), (r.xmax, r.ymax, z), (r.xmin, r.ymax, z)]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i + 1) % 4], lineColorRGB=[0.1, 0.9, 0.1], lineWidth=2, lifeTime=0)
        p.addUserDebugText(r.name, [r.drop_xy[0], r.drop_xy[1], 0.02], textColorRGB=[1, 1, 1], lifeTime=0)

def parse_triplet(text_value: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text_value.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return default

def build_world_signature(
    *,
    ycb_repo_dir: str,
    ycb_objects: List[str],
    ycb_positions: List[Tuple[float, float, float]],
    onto_uri: str,
    iri_object: str,
    iri_room: str,
    yolo_weights: str,
    jso800_urdf: str,
    use_fixed_base: bool,
    auto_robot_base: bool,
    robot_base_pos: Tuple[float, float, float],
    robot_base_rpy_deg: Tuple[float, float, float],
) -> Tuple[Any, ...]:
    norm_positions = tuple(tuple(float(v) for v in pos) for pos in ycb_positions)
    return (
        os.path.abspath(os.path.expanduser(ycb_repo_dir)),
        tuple(ycb_objects),
        norm_positions,
        onto_uri,
        iri_object,
        iri_room,
        os.path.abspath(os.path.expanduser(yolo_weights)),
        os.path.abspath(os.path.expanduser(jso800_urdf)),
        bool(use_fixed_base),
        bool(auto_robot_base),
        tuple(float(v) for v in robot_base_pos),
        tuple(float(v) for v in robot_base_rpy_deg),
    )

def clear_runtime_state(keep_ui_logs: bool = True) -> None:
    runtime_keys = [
        "robot_id", "sim_objects", "jmap", "ee_link", "onto_info", "hierarchy",
        "yolo", "clips_env", "cam", "initialized", "running", "last_result",
        "cosmos_model", "world_signature",
    ]
    preserved_logs = list(st.session_state.get("ui_logs", [])) if keep_ui_logs else []
    for key in runtime_keys:
        st.session_state.pop(key, None)
    if keep_ui_logs:
        st.session_state["ui_logs"] = preserved_logs

def setup_pybullet_world(
    gui: bool,
    ycb_repo_dir: str,
    ycb_objects: List[str],
    ycb_positions: List[Tuple[float, float, float]],
    *,
    jso800_urdf: str,
    use_fixed_base: bool = True,
    auto_robot_base: bool = True,
    robot_base_pos: Tuple[float, float, float] = ROBOT_BASE_POS_DEFAULT,
    robot_base_rpy_deg: Tuple[float, float, float] = ROBOT_BASE_RPY_DEG_DEFAULT,
) -> Tuple[int, Dict[str, int], Dict[str, int], int]:
    if p.isConnected():
        try:
            p.disconnect()
        except Exception:
            pass

    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.loadURDF("plane.urdf")

    urdf_path = os.path.abspath(os.path.expanduser(jso800_urdf))
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"JSO-800 URDF not found: {urdf_path}")

    urdf_dir = os.path.dirname(urdf_path)
    p.setAdditionalSearchPath(urdf_dir)

    if auto_robot_base and ycb_positions:
        ox, oy, _ = ycb_positions[0]
        robot_base_xyz = [float(ox - 0.28), float(oy - 0.30), 0.0]
    else:
        robot_base_xyz = [float(robot_base_pos[0]), float(robot_base_pos[1]), float(robot_base_pos[2])]

    robot_base_rpy_rad = [float(np.deg2rad(v)) for v in robot_base_rpy_deg]
    robot_base_quat = p.getQuaternionFromEuler(robot_base_rpy_rad)

    log(f"Loading JSO-800 URDF: {urdf_path}")
    log(f"URDF assets dir: {urdf_dir}")
    log(f"Robot base position: {robot_base_xyz}")
    log(f"Robot base orientation RPY deg: {list(robot_base_rpy_deg)}")

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=list(robot_base_xyz),
        baseOrientation=robot_base_quat,
        useFixedBase=use_fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    jmap = get_jso_joint_map(robot_id)
    missing = [jn for jn in JSO_ARM_JOINT_NAMES + [JSO_GRIPPER_JOINT_NAME] if jn not in jmap]
    if missing:
        raise RuntimeError(f"JSO-800 URDF missing joints: {missing}. Found: {sorted(jmap.keys())}")

    ee_link = jmap[JSO_GRIPPER_JOINT_NAME]
    jso_home(robot_id, jmap)

    paths = ensure_repo(ycb_repo_dir)
    available = set(list_ycb_objects(paths.ycb_dir))

    sim_objects: Dict[str, int] = {}
    for i, name in enumerate(ycb_objects):
        if name not in available:
            raise RuntimeError(f"YCB object '{name}' not found.")
        obj_dir = os.path.join(paths.ycb_dir, name)
        pos = ycb_positions[i] if i < len(ycb_positions) else (0.45 + 0.05 * i, 0.0, 0.15)
        bid = load_ycb_object(obj_dir=obj_dir, base_pos=pos)
        sim_objects[name] = bid
        log(f"YCB loaded '{name}' body_id={bid} at {pos}")

    draw_rooms_debug()
    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.15, cameraYaw=35, cameraPitch=-35, cameraTargetPosition=[0.30, 0.0, 0.18])

    return robot_id, sim_objects, jmap, ee_link


# ============================================================
# One cycle
# ============================================================

def run_cycle(robot_id: int, sim_objects: Dict[str, int], cam: CameraConfig, yolo, onto_info: Dict[str, OntoIndexEntry], hierarchy: Dict[str, List[str]], clips_env,
              *, use_cosmos: bool, cosmos_model: str, cosmos_user_task: str, cosmos_max_new_tokens: int,
              yolo_conf: float, yolo_to_onto: Dict[str, str], fallback_sim: bool, apply_execution: bool,
              jmap: Dict[str, int], ee_link: int) -> Dict[str, Any]:
    rgb, view, proj = render_camera(cam)
    results = yolo.predict(source=rgb, conf=float(yolo_conf), verbose=False)

    detections: List[Dict[str, Any]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = yolo.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            body_id = associate_detection_to_body((x1, y1, x2, y2), sim_objects, view, proj, cam.width, cam.height)
            if body_id is None:
                continue

            onto_key = yolo_to_onto.get(cls_name, cls_name).lower()
            if onto_key not in onto_info:
                continue

            entry = onto_info[onto_key]
            (px, py, _), _ = p.getBasePositionAndOrientation(body_id)
            cur_room = room_from_xy(px, py) or "unknown"

            detections.append({
                "source": "yolo",
                "yolo": cls_name,
                "onto": onto_key,
                "body_id": int(body_id),
                "cur_room": cur_room,
                "dst_room": (entry.room.lower() if entry.room else "none"),
                "pickable": bool(entry.pickable),
                "conf": float(box.conf[0]) if hasattr(box, "conf") else 0.0,
                "bbox": [x1, y1, x2, y2],
            })

    if fallback_sim and len(detections) == 0:
        detections = synthesize_detections_from_sim(sim_objects, onto_info)

    clear_clips_dynamic(clips_env)
    for d in detections:
        sym_id = body_symbol(int(d["body_id"]))
        pickable_sym = "yes" if d["pickable"] else "no"
        clips_env.assert_string(f'(obj (id {sym_id}) (onto "{d["onto"]}") (pickable {pickable_sym}) (cur_room {d["cur_room"]}) (dst_room {d["dst_room"]}))')

    cosmos_raw = ""
    cosmos_parsed: Dict[str, Any] = {}
    cosmos_intents: List[Dict[str, Any]] = []

    if use_cosmos and len(detections) > 0:
        with tempfile.TemporaryDirectory() as td:
            img_path = str(Path(td) / "frame.png")
            save_rgb_image(rgb, img_path)

            onto_objects = ontology_objects_compact(onto_info)
            onto_edges = ontology_hierarchy_compact(hierarchy)
            perceived = [{"onto": d["onto"], "body_id": d["body_id"], "cur_room": d["cur_room"], "dst_room": d["dst_room"], "pickable": d["pickable"]} for d in detections]

            prompt = build_cosmos_highlevel_prompt(onto_objects, onto_edges, perceived, cosmos_user_task)
            cosmos_raw, cosmos_parsed, cosmos_intents = run_cosmos_highlevel_on_image(img_path, prompt, cosmos_model, int(cosmos_max_new_tokens))

        for it in cosmos_intents:
            to_room = it.get("to_room", "none") or "none"
            rationale = str(it.get("rationale", "")).replace('"', "'")
            clips_env.assert_string(f'(intent (onto "{it["onto"]}") (action {it["action"]}) (to {to_room}) (priority {int(it["priority"])}) (rationale "{rationale}"))')

    clips_env.run()
    tasks = clips_get_tasks(clips_env)
    traces = clips_get_traces(clips_env)

    exec_logs: List[str] = []
    if apply_execution:
        move_tasks = [t for t in tasks if t["action"] == "move"]
        if move_tasks:
            t0 = move_tasks[0]
            bid = symbol_to_body(t0["id"])
            dst = t0["to"]
            if bid is not None and dst in set(ROOM_NAMES):
                exec_logs = execute_pick_and_place(robot_id, jmap, ee_link, bid, dst)

    return {
        "rgb": rgb,
        "detections": detections,
        "cosmos_raw": cosmos_raw,
        "cosmos_parsed": cosmos_parsed,
        "cosmos_intents": cosmos_intents,
        "clips_tasks": tasks,
        "clips_traces": traces,
        "exec_logs": exec_logs,
    }


# ============================================================
# Streamlit init
# ============================================================

def init_state(gui: bool, ycb_repo_dir: str, ycb_objects: List[str], ycb_positions: List[Tuple[float, float, float]],
               onto_uri: str, iri_object: str, iri_room: str, yolo_weights: str,
               jso800_urdf: str, use_fixed_base: bool = True,
               auto_robot_base: bool = True,
               robot_base_pos: Tuple[float, float, float] = ROBOT_BASE_POS_DEFAULT,
               robot_base_rpy_deg: Tuple[float, float, float] = ROBOT_BASE_RPY_DEG_DEFAULT) -> None:
    desired_signature = build_world_signature(
        ycb_repo_dir=ycb_repo_dir,
        ycb_objects=ycb_objects,
        ycb_positions=ycb_positions,
        onto_uri=onto_uri,
        iri_object=iri_object,
        iri_room=iri_room,
        yolo_weights=yolo_weights,
        jso800_urdf=jso800_urdf,
        use_fixed_base=use_fixed_base,
        auto_robot_base=auto_robot_base,
        robot_base_pos=robot_base_pos,
        robot_base_rpy_deg=robot_base_rpy_deg,
    )

    previous_signature = st.session_state.get("world_signature")
    if st.session_state.get("initialized", False) and previous_signature == desired_signature:
        return

    if st.session_state.get("initialized", False) and previous_signature != desired_signature:
        log(tr("base_pose_changed"))
        clear_runtime_state(keep_ui_logs=True)

    st.session_state.setdefault("ui_logs", [])
    log("Initializing world...")

    robot_id, sim_objects, jmap, ee_link = setup_pybullet_world(
        gui, ycb_repo_dir, ycb_objects, ycb_positions,
        jso800_urdf=jso800_urdf,
        use_fixed_base=use_fixed_base,
        auto_robot_base=auto_robot_base,
        robot_base_pos=robot_base_pos,
        robot_base_rpy_deg=robot_base_rpy_deg,
    )

    log("Loading ontology...")
    _, onto_info, hierarchy = load_ontology_index(onto_uri, iri_object, iri_room)
    log(f"Ontology loaded: objects={len(onto_info)}")

    log("Loading YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(yolo_weights)
    log("YOLO loaded")

    log("Loading CLIPS...")
    import clips
    clips_env = clips.Environment()
    build_clips_planner(clips_env)
    log("CLIPS planner loaded")

    st.session_state.update({
        "robot_id": robot_id,
        "sim_objects": sim_objects,
        "jmap": jmap,
        "ee_link": ee_link,
        "onto_info": onto_info,
        "hierarchy": hierarchy,
        "yolo": yolo,
        "clips_env": clips_env,
        "cam": CameraConfig(),
        "initialized": True,
        "running": False,
        "last_result": None,
        "world_signature": desired_signature,
    })


# ============================================================
# Main UI
# ============================================================

def main():
    st.set_page_config(page_title=tr("app_title"), layout="wide")
    st.title(tr("headline"))
    st.caption(tr("subtitle"))

    with st.sidebar:
        st.header(tr("controls"))
        st.selectbox(tr("language"), ["en", "es"], index=0 if st.session_state.get("lang","en")=="en" else 1, key="lang")
        mode = st.radio(tr("mode"), [tr("mode_single"), tr("mode_cont")], index=0)

        jso800_urdf = st.text_input(tr("urdf_path"), value=JSO800_URDF_DEFAULT)
        use_fixed_base = st.toggle(tr("use_fixed_base"), value=ROBOT_USE_FIXED_BASE_DEFAULT)
        auto_robot_base = st.toggle(tr("auto_robot_base"), value=ROBOT_AUTO_BASE_DEFAULT)
        robot_base_pos_str = st.text_input(
            tr("robot_base_pos"),
            value=",".join(map(str, ROBOT_BASE_POS_DEFAULT))
        )
        robot_base_rpy_deg_str = st.text_input(
            tr("robot_base_rpy"),
            value=",".join(map(str, ROBOT_BASE_RPY_DEG_DEFAULT))
        )

        apply_execution = st.toggle(tr("execute"), value=False)
        use_cosmos = st.toggle(tr("use_cosmos"), value=True)

        st.subheader(tr("cosmos"))
        cosmos_model = st.text_input(tr("cosmos_model"), value=COSMOS_MODEL_DEFAULT)
        cosmos_max_new_tokens = st.slider(tr("cosmos_max_new_tokens"), 64, 1024, 512, 64)
        cosmos_user_task = st.text_area(
            tr("cosmos_task"),
            value="Decide which objects should be moved to their destination rooms according to the ontology, and justify the decision.",
            height=120
        )

        st.subheader(tr("yolo"))
        yolo_weights = st.text_input(tr("yolo_weights"), value=YOLO_WEIGHTS_DEFAULT)
        yolo_conf = st.slider(tr("yolo_conf"), 0.05, 0.95, float(YOLO_CONF_DEFAULT), 0.05)
        fallback_sim = st.toggle(tr("fallback_sim"), value=True)

        st.subheader(tr("ontology"))
        onto_path = st.text_input(tr("owl_path"), value=ONTO_PATH_DEFAULT)
        onto_upload = st.file_uploader(tr("owl_upload"), type=["owl"])
        iri_object = st.text_input("IRI Object", value=IRI_OBJECT_DEFAULT)
        iri_room = st.text_input("IRI Room", value=IRI_ROOM_DEFAULT)

        st.subheader(tr("ycb"))
        ycb_repo_dir = st.text_input(tr("ycb_repo"), value=YCB_REPO_DIR_DEFAULT)
        ycb_objects_str = st.text_input(tr("ycb_objects"), value=",".join(YCB_OBJECTS_DEFAULT))
        ycb_positions_str = st.text_area(tr("ycb_positions"), value="\n".join([",".join(map(str, p)) for p in YCB_BASE_POS_DEFAULT]), height=90)

        st.subheader(tr("map_title"))
        yolo_to_onto_json = st.text_area("YOLO_TO_ONTO", value=json.dumps(YOLO_TO_ONTO_DEFAULT, ensure_ascii=False, indent=2), height=170)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(tr("reset")):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        with col2:
            if st.button(tr("step")):
                st.session_state["running"] = False
                st.session_state["do_step"] = True

        if mode == tr("mode_cont"):
            st.session_state["running"] = st.toggle(tr("run"), value=st.session_state.get("running", False))

    resolved_onto = onto_path
    if onto_upload is not None:
        up_dir = Path(tempfile.gettempdir()) / "ontoai_uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = up_dir / onto_upload.name
        with open(resolved_path, "wb") as f:
            f.write(onto_upload.getbuffer())
        resolved_onto = f"file://{resolved_path}"
    else:
        if resolved_onto and not resolved_onto.startswith(("file://", "http://", "https://")):
            resolved_onto = f"file://{resolved_onto}"

    ycb_objects = [x.strip() for x in ycb_objects_str.split(",") if x.strip()]
    ycb_positions = []
    for line in ycb_positions_str.strip().splitlines():
        parts = [pp.strip() for pp in line.split(",") if pp.strip()]
        if len(parts) != 3:
            continue
        try:
            ycb_positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except Exception:
            pass
    if not ycb_positions:
        ycb_positions = list(YCB_BASE_POS_DEFAULT)

    robot_base_pos = parse_triplet(robot_base_pos_str, ROBOT_BASE_POS_DEFAULT)
    robot_base_rpy_deg = parse_triplet(robot_base_rpy_deg_str, ROBOT_BASE_RPY_DEG_DEFAULT)

    init_state(
        True,
        ycb_repo_dir,
        ycb_objects,
        ycb_positions,
        resolved_onto,
        iri_object,
        iri_room,
        yolo_weights,
        jso800_urdf,
        use_fixed_base,
        auto_robot_base,
        robot_base_pos,
        robot_base_rpy_deg
    )

    try:
        yolo_to_onto = json.loads(yolo_to_onto_json)
        if not isinstance(yolo_to_onto, dict):
            yolo_to_onto = dict(YOLO_TO_ONTO_DEFAULT)
    except Exception:
        yolo_to_onto = dict(YOLO_TO_ONTO_DEFAULT)

    do_step = st.session_state.pop("do_step", False)
    if st.session_state.get("running", False) or do_step:
        res = run_cycle(
            robot_id=st.session_state["robot_id"],
            sim_objects=st.session_state["sim_objects"],
            cam=st.session_state["cam"],
            yolo=st.session_state["yolo"],
            onto_info=st.session_state["onto_info"],
            hierarchy=st.session_state["hierarchy"],
            clips_env=st.session_state["clips_env"],
            use_cosmos=use_cosmos,
            cosmos_model=cosmos_model,
            cosmos_user_task=cosmos_user_task,
            cosmos_max_new_tokens=cosmos_max_new_tokens,
            yolo_conf=yolo_conf,
            yolo_to_onto=yolo_to_onto,
            fallback_sim=fallback_sim,
            apply_execution=apply_execution,
            jmap=st.session_state["jmap"],
            ee_link=st.session_state["ee_link"],
        )
        st.session_state["last_result"] = res

    res = st.session_state.get("last_result")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader(tr("camera"))
        if res is not None:
            st.image(res["rgb"], channels="RGB", width="stretch")
        else:
            st.info(tr("press_to_run"))

        st.subheader(tr("detections"))
        if res is not None:
            if res["detections"]:
                st.dataframe(res["detections"], width="stretch", hide_index=True)
            else:
                st.warning(tr("no_detections"))

        st.subheader(tr("logs"))
        st.text("\n".join(st.session_state.get("ui_logs", [])[-120:]))

    with right:
        st.subheader(tr("cosmos_reasoner"))
        if res is not None and use_cosmos:
            st.text_area(tr("raw_output"), value=res.get("cosmos_raw", ""), height=160)
            st.markdown(tr("parsed_json"))
            st.json(res.get("cosmos_parsed", {}))
            st.markdown(tr("intents"))
            st.dataframe(res.get("cosmos_intents", []), width="stretch", hide_index=True)
        else:
            st.caption(tr("enable_cosmos_hint"))

        st.subheader(tr("clips_planner"))
        if res is not None:
            st.dataframe(res.get("clips_tasks", []), width="stretch", hide_index=True)
            st.text("\n".join(res.get("clips_traces", [])) if res.get("clips_traces") else "")

        st.subheader(tr("execution"))
        if res is not None and res.get("exec_logs"):
            st.success(tr("executed"))
            st.text("\n".join(res["exec_logs"]))
        else:
            if apply_execution and res is not None and (not any(t.get("action") == "move" for t in res.get("clips_tasks", []))):
                st.warning(tr("no_move"))
            st.caption(tr("enable_execute_hint"))

    if st.session_state.get("running", False):
        time.sleep(0.15)
        st.rerun()


if __name__ == "__main__":
    main()