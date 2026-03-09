#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGV + mock Cosmos-Reasoner2 + ontology-based object reorganization demo.

What this demo does
-------------------
- Builds 4 places (kitchen, bathroom, living_room, bedroom) in PyBullet.
- Places one cube pedestal per room.
- Places several objects on the wrong pedestals.
- Uses a small ontology layer to know where each object belongs.
- Uses a functional "mock Cosmos-Reasoner2" adapter that returns structured JSON
  from the robot's local scene.
- Drives a simple UGV through the environment, picks objects, and reorganizes them
  according to the ontology.

Install:
    pip install pybullet numpy

Run:
    python ugv_cosmos_reasoner2_reorg_demo.py --gui --realtime_sleep 0.004

This version is fully functional today because the Reasoner2 module is mocked but
keeps the same contract you would use with a real visual-semantic model:
    image/state/mission -> JSON
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


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


# -----------------------------------------------------------------------------
# Ontology layer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticObject:
    name: str
    category: str
    home_room: str
    color_rgba: Tuple[float, float, float, float]


class HouseOntology:
    def __init__(self) -> None:
        self.rooms = ["kitchen", "bathroom", "living_room", "bedroom"]
        self.objects: Dict[str, SemanticObject] = {
            "mug_red": SemanticObject("mug_red", "mug", "kitchen", (0.85, 0.15, 0.15, 1.0)),
            "soap_blue": SemanticObject("soap_blue", "soap", "bathroom", (0.15, 0.35, 0.95, 1.0)),
            "remote_black": SemanticObject("remote_black", "remote", "living_room", (0.15, 0.15, 0.15, 1.0)),
            "pillow_green": SemanticObject("pillow_green", "pillow", "bedroom", (0.15, 0.8, 0.25, 1.0)),
        }

    def target_room_for(self, object_name: str) -> str:
        return self.objects[object_name].home_room

    def object_names(self) -> List[str]:
        return list(self.objects.keys())

    def export_turtle(self, path: str) -> None:
        ttl = [
            "@prefix ex: <http://example.org/home#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "",
            "ex:Room a owl:Class .",
            "ex:Object a owl:Class .",
            "ex:belongsIn a owl:ObjectProperty .",
            "",
        ]
        for room in self.rooms:
            ttl.append(f"ex:{room} a ex:Room .")
        ttl.append("")
        for obj in self.objects.values():
            ttl.append(f"ex:{obj.name} a ex:Object ; ex:belongsIn ex:{obj.home_room} .")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ttl))


# -----------------------------------------------------------------------------
# World
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


class HouseWorld:
    def __init__(self, client_id: int, ontology: HouseOntology):
        self.cid = client_id
        self.ontology = ontology
        self.room_size = 2.8
        self.rooms: Dict[str, RoomInfo] = {}
        self.objects: Dict[str, SimObjectState] = {}
        self._build_world()

    def _create_box(self, half_extents, mass, pos, rgba, collision=True):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.cid) if collision else -1
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=self.cid,
        )

    def _build_world(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        room_defs = [
            ("kitchen", (-2.0, 2.0), (0.95, 0.85, 0.70, 1.0)),
            ("bathroom", (2.0, 2.0), (0.75, 0.88, 0.98, 1.0)),
            ("living_room", (-2.0, -2.0), (0.78, 0.92, 0.78, 1.0)),
            ("bedroom", (2.0, -2.0), (0.93, 0.80, 0.92, 1.0)),
        ]

        for name, center_xy, floor_rgba in room_defs:
            cx, cy = center_xy
            self._create_box([self.room_size / 2, self.room_size / 2, 0.01], 0, [cx, cy, 0.0], floor_rgba, collision=False)
            pedestal_pos = (cx, cy, 0.20)
            pedestal_id = self._create_box([0.18, 0.18, 0.20], 0, pedestal_pos, (0.65, 0.65, 0.65, 1.0), collision=True)
            self.rooms[name] = RoomInfo(name, center_xy, floor_rgba, pedestal_pos, pedestal_id)

        self._build_walls()
        self._spawn_misplaced_objects()

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

    def _spawn_misplaced_objects(self) -> None:
        wrong_pedestal_for = {
            "mug_red": "bathroom",
            "soap_blue": "living_room",
            "remote_black": "bedroom",
            "pillow_green": "kitchen",
        }
        for obj_name, sem_obj in self.ontology.objects.items():
            wrong_room = wrong_pedestal_for[obj_name]
            px, py, pz = self.rooms[wrong_room].pedestal_pos
            body_id = self._create_box([0.06, 0.06, 0.06], 0.15, [px, py, pz + 0.19], sem_obj.color_rgba, collision=True)
            self.objects[obj_name] = SimObjectState(obj_name, body_id, wrong_room, wrong_room, False)

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
        return all(self.objects[name].current_room == self.ontology.target_room_for(name) for name in self.objects)

    def pedestal_drop_pose(self, room_name: str) -> Tuple[float, float, float]:
        px, py, pz = self.rooms[room_name].pedestal_pos
        return px, py, pz + 0.24

    def object_pose(self, object_name: str) -> Tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(self.objects[object_name].body_id, physicsClientId=self.cid)
        return float(pos[0]), float(pos[1]), float(pos[2])


# -----------------------------------------------------------------------------
# Mock Cosmos-Reasoner2
# -----------------------------------------------------------------------------

class MockCosmosReasoner2:
    def __init__(self, world: HouseWorld):
        self.world = world

    def reason(self, robot_xyyaw: Tuple[float, float, float], mission: str) -> Dict:
        rx, ry, ryaw = robot_xyyaw
        visible = []
        hazards = []

        for obj_name in self.world.objects:
            ox, oy, _ = self.world.object_pose(obj_name)
            d = distance2d((rx, ry), (ox, oy))
            bearing = math.atan2(oy - ry, ox - rx)
            rel = wrap_angle(bearing - ryaw)
            if d < 2.4 and abs(rel) < math.radians(70):
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
        return {
            "scene_summary": f"Robot is in {current_room} and sees {len(visible)} relevant objects.",
            "current_room": current_room,
            "visible_objects": visible,
            "hazards": hazards,
            "recommended_next_action": "inspect_visible_objects" if visible else "continue_navigation",
            "mission": mission,
        }


# -----------------------------------------------------------------------------
# UGV
# -----------------------------------------------------------------------------

class SimpleUGV:
    def __init__(self, client_id: int, start_pos=(0.0, 0.0, 0.12), start_yaw=0.0):
        self.cid = client_id
        self.body_id = self._build_robot(start_pos, start_yaw)
        self.carry_height = 0.30
        self.max_lin = 1.0
        self.max_ang = 2.0

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
        p.resetBaseVelocity(self.body_id, linearVelocity=[vx, vy, 0.0], angularVelocity=[0.0, 0.0, ang], physicsClientId=self.cid)

    def stop(self) -> None:
        self._set_velocity(0.0, 0.0)

    def step_to_pose(self, target_xy: Tuple[float, float], dt: float = 1.0 / 120.0) -> Tuple[float, float]:
        x, y, yaw = self.pose()
        dx = target_xy[0] - x
        dy = target_xy[1] - y
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        yaw_err = wrap_angle(desired - yaw)
        lin = clamp(0.9 * dist, -self.max_lin, self.max_lin)
        ang = clamp(2.0 * yaw_err, -self.max_ang, self.max_ang)
        if abs(yaw_err) > 0.7:
            lin = 0.0
        self._set_velocity(lin, ang)
        p.stepSimulation(physicsClientId=self.cid)
        return dist, yaw_err

    def hold_object(self, object_body_id: int) -> None:
        x, y, yaw = self.pose()
        hold_pos = [x + 0.23 * math.cos(yaw), y + 0.23 * math.sin(yaw), self.carry_height]
        p.resetBasePositionAndOrientation(object_body_id, hold_pos, quat_from_yaw(yaw), physicsClientId=self.cid)
        p.resetBaseVelocity(object_body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)


# -----------------------------------------------------------------------------
# Planner / executor
# -----------------------------------------------------------------------------

class OntologyPlanner:
    def __init__(self, world: HouseWorld, ontology: HouseOntology, ugv: SimpleUGV, reasoner: MockCosmosReasoner2):
        self.world = world
        self.ontology = ontology
        self.ugv = ugv
        self.reasoner = reasoner
        self.current_carried: Optional[str] = None

    def plan_next_task(self) -> Optional[Dict]:
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

    def execute_task(self, task: Dict, realtime_sleep: float = 0.0) -> None:
        obj_name = task["object"]
        src_room = task["source_room"]
        dst_room = task["target_room"]

        print("\n" + "=" * 80)
        print(f"TASK: move {obj_name} from {src_room} to {dst_room}")
        print("=" * 80)

        src_xy = self.world.rooms[src_room].pedestal_pos[:2]
        self.goto_xy(src_xy, label=f"go to source pedestal in {src_room}", realtime_sleep=realtime_sleep)

        self.world.update_object_rooms()
        reason_json = self.reasoner.reason(self.ugv.pose(), mission=f"Find {obj_name} and reorganize it")
        print("[REASONER]", json.dumps(reason_json, indent=2))

        visible_names = [v["name"] for v in reason_json["visible_objects"]]
        if obj_name not in visible_names:
            self.local_scan(obj_name, realtime_sleep=realtime_sleep)
            reason_json = self.reasoner.reason(self.ugv.pose(), mission=f"Find {obj_name} and reorganize it")
            print("[REASONER AFTER SCAN]", json.dumps(reason_json, indent=2))

        obj_pos = self.world.object_pose(obj_name)
        self.goto_xy((obj_pos[0], obj_pos[1]), stop_radius=0.18, label=f"approach {obj_name}", realtime_sleep=realtime_sleep)
        self.pick_object(obj_name, realtime_sleep=realtime_sleep)

        dst_xy = self.world.rooms[dst_room].pedestal_pos[:2]
        self.goto_xy(dst_xy, label=f"carry {obj_name} to {dst_room}", realtime_sleep=realtime_sleep)
        self.drop_object(obj_name)

        self.world.update_object_rooms()
        print(f"[DONE] {obj_name} now in room: {self.world.objects[obj_name].current_room}")

    def goto_xy(self, xy: Tuple[float, float], stop_radius: float = 0.10, label: str = "goto", realtime_sleep: float = 0.0, timeout_s: float = 25.0) -> None:
        print(f"[NAV] {label} -> target={tuple(round(v, 3) for v in xy)}")
        dt = 1.0 / 120.0
        max_steps = int(timeout_s / dt)
        for step in range(max_steps):
            dist, yaw_err = self.ugv.step_to_pose(xy, dt=dt)
            if self.current_carried is not None:
                self.ugv.hold_object(self.world.objects[self.current_carried].body_id)
            if step % 60 == 0:
                x, y, yaw = self.ugv.pose()
                print(f"  step={step:04d} pose=({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg) dist={dist:.2f} yaw_err={math.degrees(yaw_err):.1f} deg")
            if dist < stop_radius:
                self.ugv.stop()
                for _ in range(25):
                    p.stepSimulation(physicsClientId=self.world.cid)
                    if self.current_carried is not None:
                        self.ugv.hold_object(self.world.objects[self.current_carried].body_id)
                    if realtime_sleep > 0:
                        time.sleep(realtime_sleep)
                return
            if realtime_sleep > 0:
                time.sleep(realtime_sleep)
        self.ugv.stop()
        print(f"[WARN] navigation timeout at target {xy}")

    def local_scan(self, target_object: str, realtime_sleep: float = 0.0) -> None:
        print(f"[SCAN] looking for {target_object}")
        for _ in range(160):
            self.ugv._set_velocity(0.0, 1.1)
            p.stepSimulation(physicsClientId=self.world.cid)
            if realtime_sleep > 0:
                time.sleep(realtime_sleep)
            reason_json = self.reasoner.reason(self.ugv.pose(), mission=f"Find {target_object}")
            visible_names = [v["name"] for v in reason_json["visible_objects"]]
            if target_object in visible_names:
                print(f"[SCAN] {target_object} found")
                self.ugv.stop()
                return
        self.ugv.stop()
        print(f"[SCAN] {target_object} not found, continuing anyway")

    def pick_object(self, object_name: str, realtime_sleep: float = 0.0) -> None:
        print(f"[PICK] {object_name}")
        self.current_carried = object_name
        for _ in range(60):
            self.ugv.hold_object(self.world.objects[object_name].body_id)
            p.stepSimulation(physicsClientId=self.world.cid)
            if realtime_sleep > 0:
                time.sleep(realtime_sleep)

    def drop_object(self, object_name: str) -> None:
        target_room = self.ontology.target_room_for(object_name)
        drop_pos = self.world.pedestal_drop_pose(target_room)
        p.resetBasePositionAndOrientation(self.world.objects[object_name].body_id, drop_pos, [0, 0, 0, 1], physicsClientId=self.world.cid)
        p.resetBaseVelocity(self.world.objects[object_name].body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.world.cid)
        self.current_carried = None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UGV + ontology + mock Cosmos-Reasoner2 demo in PyBullet")
    ap.add_argument("--gui", action="store_true", help="Launch PyBullet with GUI")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--realtime_sleep", type=float, default=0.0, help="Sleep per sim step for visual debugging, e.g. 0.004")
    ap.add_argument("--export_ontology", default="house_ontology.ttl", help="Path to export a tiny Turtle ontology")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(
        cameraDistance=8.8,
        cameraYaw=40,
        cameraPitch=-58,
        cameraTargetPosition=[0.0, 0.0, 0.0],
        physicsClientId=cid,
    )

    ontology = HouseOntology()
    ontology.export_turtle(args.export_ontology)
    world = HouseWorld(cid, ontology)
    ugv = SimpleUGV(cid, start_pos=(0.0, 0.0, 0.12), start_yaw=0.0)
    reasoner = MockCosmosReasoner2(world)
    planner = OntologyPlanner(world, ontology, ugv, reasoner)

    print("Ontology exported to:", args.export_ontology)
    print("Initial object locations:")
    world.update_object_rooms()
    for obj_name in ontology.object_names():
        st = world.objects[obj_name]
        print(f"  - {obj_name:14s} in {st.current_room:12s} -> should be in {ontology.target_room_for(obj_name)}")

    while True:
        task = planner.plan_next_task()
        if task is None:
            break
        planner.execute_task(task, realtime_sleep=args.realtime_sleep)

    print("\nFinal object locations:")
    world.update_object_rooms()
    for obj_name in ontology.object_names():
        st = world.objects[obj_name]
        print(f"  - {obj_name:14s} in {st.current_room:12s} -> target {ontology.target_room_for(obj_name)}")

    print("\nSorted correctly:", world.all_objects_sorted())
    print("Demo finished.")

    if args.gui:
        print("Close the PyBullet window to exit.")
        try:
            while p.isConnected(physicsClientId=cid):
                p.stepSimulation(physicsClientId=cid)
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            pass

    p.disconnect(cid)


if __name__ == "__main__":
    main()

# pip install pybullet numpy
# python -m src.ugv.ugv_cosmos_reasoner2_reorg_demo --gui --realtime_sleep 0.004