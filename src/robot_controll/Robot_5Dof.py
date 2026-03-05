import json
import time
from threading import Thread
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from pylx16a.lx16a import LX16A, ServoTimeoutError

JointName = str
ServoSpec = Tuple[int, str]  # (id, "lx16a")


class Robot5Dof(Thread):
    """
    Control 5DOF LX16A con configuración por JSON.

    Soporta 2 escenarios:
    - mode="teleop": lee leader (si existe) y aplica a followers (uno o varios).
    - mode="eval": NO requiere leader. Solo configura followers y expone apply().

    Soporta:
    - Un roll o varios: arm_rolls="arm_roll" o arm_rolls=["arm_roll", "arm_roll_2"]
    - Un follower o varios: follower_roles=["follower_a"] o ["follower_a","follower_b"]

    ✅ MOD: Interpolación lineal en el "write" (apply):
    - En lugar de un salto directo al target, envía varios pasos lineales desde el ángulo actual.
    - Útil para suavizar movimientos y reducir tirones/overshoot.
    """

    DEFAULT_5DOF_JOINTS: List[JointName] = ["joint_base", "joint_1", "joint_2", "joint_4", "joint_5"]

    def __init__(
        self,
        usb_port: str = "/dev/ttyUSB0",
        config_path: str = "arm_config.json",

        # Permite 1 roll o varios
        arm_rolls: Union[str, List[str]] = "arm_roll",

        # Teleop: leader opcional (en eval no se usa)
        leader_role: Optional[str] = "leader",

        # Permite 1 follower o varios
        follower_roles: Union[str, List[str]] = "follower_a",

        # "teleop" o "eval"
        mode: str = "teleop",

        slave_speed: int = 85,
        log_path: str = "LeaderAngles.json",
        initialization_angles: Optional[List[float]] = None,

        iterations: int = 200,
        delay: float = 0.03,
        joints_5dof: Optional[List[JointName]] = None,
        verbose: bool = False,

        # -------------------------
        # ✅ Interpolación (nuevo)
        # -------------------------
        interp_enable: bool = True,
        interp_step_ms: int = 29,              # tiempo entre pasos (ms). 20-30ms suele ir bien.
        interp_min_steps: int = 3,             # mínimo pasos si interp activa
        interp_max_steps: int = 30,            # evita spamear el bus
        interp_use_feedback: bool = True,      # lee ángulo físico al inicio (más suave, más lento)
        interp_deadband_deg: float = 0.25,     # no mover si cambio es mínimo
        # -------------------------
        # ✅ Suavizado extra (nuevo)
        # -------------------------
        interp_filter_beta: float = 0.20,     # low-pass sobre target (0..1)
        interp_max_speed_deg_s: float = 180.0,# limitador de velocidad (deg/s)
        interp_limit_eps_deg: float = 0.50,   # margen para no tocar límites estrictos de pylx16a
    ):
        super().__init__(daemon=True)

        self.usb_port = usb_port
        self.config_path = config_path

        self.arm_rolls = [arm_rolls] if isinstance(arm_rolls, str) else list(arm_rolls)
        self.follower_roles = [follower_roles] if isinstance(follower_roles, str) else list(follower_roles)

        self.mode = str(mode).lower().strip()
        if self.mode not in ("teleop", "eval"):
            raise ValueError("mode debe ser 'teleop' o 'eval'.")

        self.leader_role = leader_role  # puede ser None en eval
        self.slave_speed = int(slave_speed)
        self.log_path = log_path
        self.verbose = bool(verbose)

        self.iterations = int(iterations)
        self.delay = float(delay)
        self.running = False

        self.joints = joints_5dof or self.DEFAULT_5DOF_JOINTS
        self.initialization_angles = initialization_angles or [180, 180, 180, 90, 70]

        self.angles_log: List[dict] = []

        # servos[roll][role][joint] = LX16A(...)
        self.servos: Dict[str, Dict[str, Dict[JointName, LX16A]]] = {}

        # ✅ estado último comando (para interpolar sin feedback si quieres)
        # last_cmd[roll][role][joint] = float(deg)
        self.last_cmd: Dict[str, Dict[str, Dict[JointName, float]]] = {}

        # Interpolación (config)
        self.interp_enable = bool(interp_enable)
        self.interp_step_ms = int(interp_step_ms)
        self.interp_min_steps = int(interp_min_steps)
        self.interp_max_steps = int(interp_max_steps)
        self.interp_use_feedback = bool(interp_use_feedback)
        self.interp_deadband_deg = float(interp_deadband_deg)
        self.interp_filter_beta = float(interp_filter_beta)
        self.interp_max_speed_deg_s = float(interp_max_speed_deg_s)
        self.interp_limit_eps_deg = float(interp_limit_eps_deg)

        # ✅ target filtrado (por roll/role/joint) para evitar temblores
        self.filt_target: Dict[str, Dict[str, Dict[JointName, float]]] = {}

        if self.interp_step_ms <= 0:
            raise ValueError("interp_step_ms debe ser > 0")

        self._init_bus()
        self._load_and_configure_servos()

    # -------------------------
    # Init / JSON
    # -------------------------
    def _init_bus(self):
        try:
            LX16A.initialize(self.usb_port)
        except Exception:
            pass  # ya inicializado

    def _load_json(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_role_block(self, cfg: dict, roll: str, role: str) -> Dict[JointName, ServoSpec]:
        try:
            return cfg[roll][role]
        except KeyError as e:
            raise KeyError(f"No existe cfg['{roll}']['{role}'] en {self.config_path}. Error: {e}")

    def _make_servo(self, servo_id: int) -> LX16A:
        try:
            return LX16A(int(servo_id))
        except ServoTimeoutError as e:
            raise ServoTimeoutError(f"El servo {e.id_} no responde (timeout).")

    def _load_and_configure_servos(self):
        cfg = self._load_json()

        # Crear estructura
        for roll in self.arm_rolls:
            self.servos.setdefault(roll, {})
            self.last_cmd.setdefault(roll, {})

            # Configurar leader solo si aplica (teleop y leader_role no es None)
            if self.mode == "teleop":
                if not self.leader_role:
                    raise ValueError("En mode='teleop' necesitas leader_role (p.ej. 'leader').")
                leader_map = self._get_role_block(cfg, roll, self.leader_role)
                self._instantiate_role_servos(roll, self.leader_role, leader_map)

            # Configurar followers (uno o varios)
            for fr in self.follower_roles:
                follower_map = self._get_role_block(cfg, roll, fr)
                self._instantiate_role_servos(roll, fr, follower_map)

        time.sleep(0.8)

    def _instantiate_role_servos(self, roll: str, role: str, role_map: Dict[JointName, ServoSpec]):
        missing = [j for j in self.joints if j not in role_map]
        if missing:
            raise ValueError(f"En roll='{roll}', role='{role}' faltan joints: {missing}")

        self.servos[roll].setdefault(role, {})
        self.last_cmd[roll].setdefault(role, {})
        self.filt_target.setdefault(roll, {})
        self.filt_target[roll].setdefault(role, {})

        for j in self.joints:
            sid, stype = role_map[j]
            if stype != "lx16a":
                raise ValueError(f"Tipo no soportado en roll='{roll}', role='{role}', joint='{j}': {stype}")
            self.servos[roll][role][j] = self._make_servo(sid)

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _clamp_deg(x: float, eps: float = 0.0) -> float:
        """Clamp degrees to safe servo limits.
        Note: pylx16a enforces strict internal limits slightly inside [0,180].
        Use eps>0 to avoid hitting boundaries exactly.
        """
        lo = 0.0 + float(eps)
        hi = 180.0 - float(eps)
        if hi <= lo:
            lo, hi = 0.0, 180.0
        return max(lo, min(hi, float(x)))

    def disable_torque(self, roll: Optional[str] = None, role: Optional[str] = None):
        """
        Deshabilita torque.
        - Si no pasas roll/role => deshabilita todo lo configurado.
        - Si pasas roll y role => solo ese conjunto.
        """
        if roll is None and role is None:
            for r in self.servos:
                print(f"Servos: {r}")
                for ro in self.servos[r]:
                    for s in self.servos[r][ro].values():
                        try:
                            s.disable_torque()
                        except Exception:
                            pass
            return

        if roll is None or role is None:
            raise ValueError("Si especificas roll o role, debes especificar ambos.")

        for s in self.servos[roll][role].values():
            try:
                s.disable_torque()
            except Exception:
                pass


    def enable_torque(self, roll: Optional[str] = None, role: Optional[str] = None):
        """
        Deshabilita torque.
        - Si no pasas roll/role => deshabilita todo lo configurado.
        - Si pasas roll y role => solo ese conjunto.
        """
        if roll is None and role is None:
            for r in self.servos:
                for ro in self.servos[r]:
                    for s in self.servos[r][ro].values():
                        try:
                            s.enable_torque()
                        except Exception:
                            pass
            return

        if roll is None or role is None:
            raise ValueError("Si especificas roll o role, debes especificar ambos.")

        for s in self.servos[roll][role].values():
            try:
                s.enable_torque()
            except Exception:
                pass

    # -------------------------
    # Read
    # -------------------------
    def read_angles(self, roll: str, role: str) -> Tuple[List[float], np.ndarray]:
        deg = []
        for j in self.joints:
            a = self.servos[roll][role][j].get_physical_angle()
            deg.append(float(a))
        rad = np.radians(deg)

        if self.verbose:
            print({f"{roll}:{role}": dict(zip(self.joints, deg))})

        return deg, rad

    def _get_start_deg_for_interp(self, roll: str, role: str) -> List[float]:
        """
        Ángulos de partida para interpolación:
        - Si interp_use_feedback: lee físico 1 vez.
        - Si no: usa last_cmd si existe; si no existe, cae a lectura física.
        """
        if self.interp_use_feedback:
            start_deg, _ = self.read_angles(roll, role)
            return start_deg

        # sin feedback: usa último comando si existe (más rápido)
        start = []
        for j in self.joints:
            if j in self.last_cmd.get(roll, {}).get(role, {}):
                start.append(float(self.last_cmd[roll][role][j]))
            else:
                # fallback seguro
                a = self.servos[roll][role][j].get_physical_angle()
                start.append(float(a))
        return start

    # -------------------------
    # Apply / Write (con interpolación)
    # -------------------------
    def apply(self, target_deg: List[float], roll: str, role: str, speed: Optional[int] = None):
        """
        Aplica ángulos a un role concreto (normalmente follower_*).

        ✅ Suavizado mejorado:
        - Low-pass sobre el target (reduce temblor por ruido de IK/VR).
        - Slew-rate limiter (deg/s) para evitar saltos bruscos.
        - Envía comandos en pasos a dt fijo (interp_step_ms), con float (sin int()).
        - Mantiene margen 'eps' para no tocar límites estrictos de pylx16a.
        """
        if len(target_deg) != len(self.joints):
            raise ValueError(f"target_deg len={len(target_deg)} pero joints={len(self.joints)} -> {self.joints}")

        eps = float(self.interp_limit_eps_deg)

        # Target clamped (con margen)
        target = [self._clamp_deg(x, eps=eps) for x in target_deg]

        total_ms = int(self.slave_speed if speed is None else speed)
        total_ms = max(1, total_ms)

        # Inicializa last_cmd y filt_target si hace falta
        start = self._get_start_deg_for_interp(roll, role)
        for j, s in zip(self.joints, start):
            self.last_cmd.setdefault(roll, {}).setdefault(role, {})
            self.last_cmd[roll][role].setdefault(j, float(s))
            self.filt_target.setdefault(roll, {}).setdefault(role, {})
            self.filt_target[roll][role].setdefault(j, float(s))

        # Sin interpolación: aplica low-pass + margen y manda float
        if not self.interp_enable:
            beta = float(np.clip(self.interp_filter_beta, 0.0, 1.0))
            for j, t in zip(self.joints, target):
                prev = float(self.filt_target[roll][role].get(j, t))
                tf = (1.0 - beta) * prev + beta * float(t)
                tf = self._clamp_deg(tf, eps=eps)
                self.servos[roll][role][j].move(float(tf), time=total_ms)
                self.last_cmd[roll][role][j] = float(tf)
                self.filt_target[roll][role][j] = float(tf)
            return

        # dt de control
        step_ms = int(self.interp_step_ms)
        step_ms = max(5, step_ms)  # evita spamear demasiado
        steps = int(np.ceil(total_ms / step_ms))
        steps = max(self.interp_min_steps, min(self.interp_max_steps, steps))

        # Limitador de velocidad (deg/s) -> deg por paso
        max_speed = max(1.0, float(self.interp_max_speed_deg_s))
        dt_s = step_ms / 1000.0
        max_step = max_speed * dt_s

        # Low-pass target (reduce jitter)
        beta = float(np.clip(self.interp_filter_beta, 0.0, 1.0))
        tgt_f = []
        for j, t in zip(self.joints, target):
            prev = float(self.filt_target[roll][role].get(j, t))
            tf = (1.0 - beta) * prev + beta * float(t)
            tf = self._clamp_deg(tf, eps=eps)
            self.filt_target[roll][role][j] = float(tf)
            tgt_f.append(float(tf))

        # Si el cambio es mínimo, evita mover (pero actualiza estados)
        max_delta = max(abs(t - float(self.last_cmd[roll][role][j])) for j, t in zip(self.joints, tgt_f))
        if max_delta < self.interp_deadband_deg:
            for j, t in zip(self.joints, tgt_f):
                self.last_cmd[roll][role][j] = float(t)
            return

        # Control por pasos: cmd <- cmd + clip(err, -max_step, +max_step)
        for _ in range(steps):
            for j, t in zip(self.joints, tgt_f):
                cmd = float(self.last_cmd[roll][role][j])
                err = float(t) - cmd
                cmd_next = cmd + float(np.clip(err, -max_step, +max_step))
                cmd_next = self._clamp_deg(cmd_next, eps=eps)

                # Enviar float (no int) para evitar escalonado a 1°
                self.servos[roll][role][j].move(float(cmd_next), time=step_ms)
                self.last_cmd[roll][role][j] = float(cmd_next)

            # pacing: respeta dt
            time.sleep(dt_s)

    def apply_all_followers(self, target_deg: List[float], speed: Optional[int] = None):
        """
        Aplica el mismo target a TODOS los followers configurados, en TODOS los rolls.
        """
        for roll in self.arm_rolls:
            for fr in self.follower_roles:
                self.apply(target_deg, roll=roll, role=fr, speed=speed)

    def save_data(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.angles_log, f, indent=2)
        print(f"Datos guardados en: {self.log_path}")

    # -------------------------
    # Thread lifecycle
    # -------------------------
    def run(self):
        """
        - teleop: lee leader y aplica a followers (en todos los rolls).
        - eval: no hace nada (usa apply/apply_all_followers desde fuera).
        """
        if self.mode == "eval":
            self.running = True
            while self.running:
                time.sleep(0.2)
            return

        # teleop
        self.running = True

        # init followers
        try:
            self.apply_all_followers(self.initialization_angles)
        except Exception:
            pass

        for _ in range(self.iterations):
            if not self.running:
                break

            for roll in self.arm_rolls:
                leader_deg, _ = self.read_angles(roll, self.leader_role)
                self.angles_log.append({f"{roll}:{self.leader_role}": leader_deg})

                for fr in self.follower_roles:
                    self.apply(leader_deg, roll=roll, role=fr)

            time.sleep(self.delay)

        self.save_data()

    def stop(self):
        self.running = False


# -------------------------
# Ejemplos de uso
# -------------------------
if __name__ == "__main__":

    # ======== 1) TELEOP (leader existe) ========
    # controller = Robot5Dof(
    #     usb_port="/dev/ttyUSB0",
    #     config_path="ACT_Main/config/lx16a_arm_config.json",
    #     arm_rolls="arm_roll",
    #     leader_role="leader",
    #     follower_roles=["follower_a"],
    #     mode="teleop",
    #     iterations=200,
    #     delay=0.03,
    #     verbose=True,
    #     interp_enable=True,
    #     interp_step_ms=25,
    #     interp_use_feedback=True,
    # )
    # controller.disable_torque(roll="arm_roll", role="leader")  # humano mueve leader
    # controller.start()
    # controller.join()

    # ======== 2) EVAL (leader NO existe) ========
    controller = Robot5Dof(
        usb_port="/dev/ttyUSB0",
        config_path="ACT_Main/config/lx16a_arm_config.json",
        arm_rolls="arm_roll",
        leader_role=None,
        follower_roles=["follower_a"],
        mode="eval",
        verbose=True,
        interp_enable=True,          # ✅ interpolación ON
        interp_step_ms=25,           # ✅ suavidad vs latencia
        interp_use_feedback=True,    # ✅ lee ángulo físico al inicio
    )

    # Ejemplo: aplicar una acción externa (p.ej. salida de tu ACTPolicy)
    action_deg = [90, 120, 80, 100, 60]
    controller.apply(action_deg, roll="arm_roll", role="follower_a", speed=120)  # 120ms total aprox

    print("Eval: acción aplicada.")



'''
    # -------------------------
    # Ejemplos de uso
    # -------------------------
    if __name__ == "__main__":

        # ======== 1) TELEOP (leader existe) ========
        # controller = Robot5Dof(
        #     usb_port="/dev/ttyUSB0",
        #     config_path="ACT_Main/config/lx16a_arm_config.json",
        #     arm_rolls="arm_roll",
        #     leader_role="leader",
        #     follower_roles=["follower_a"],
        #     mode="teleop",
        #     iterations=200,
        #     delay=0.03,
        #     verbose=True,
        # )
        # controller.disable_torque(roll="arm_roll", role="leader")  # humano mueve leader
        # controller.start()
        # controller.join()

        # ======== 2) EVAL (leader NO existe) ========
        controller = Robot5Dof(
            usb_port="/dev/ttyUSB0",
            config_path="ACT_Main/config/lx16a_arm_config.json",
            arm_rolls="arm_roll",                 # o ["arm_roll","arm_roll_2"]
            leader_role=None,                     # importante
            follower_roles=["follower_a"],         # o ["follower_a","follower_b"]
            mode="eval",
            verbose=True,
        )

        # En evaluación: no uses disable_torque sobre el follower si quieres que actúe.
        # Si quieres soltarlos para mover a mano:
        # controller.disable_torque(roll="arm_roll", role="follower_a")

        # Ejemplo: aplicar una acción externa (p.ej. salida de tu ACTPolicy)
        action_deg = [90, 120, 80, 100, 60]  # 5 DOF en el orden joints
        controller.apply(action_deg, roll="arm_roll", role="follower_a", speed=85)

        print("Eval: acción aplicada.")




    # Cómo lo usas en tu evaluación (sin leader)

    Tu loop de evaluación típicamente hace:


    controller = Robot5Dof(
        config_path=".../lx16a_arm_config.json",
        arm_rolls="arm_roll",           # o lista de rolls
        follower_roles=["follower_a"],  # o varios
        leader_role=None,
        mode="eval"
    )

    # Cada paso:
    pred_deg = model_output_deg  # [5]
    controller.apply_all_followers(pred_deg)  # o apply(pred_deg, roll=..., role=...)


    # Importante: tu llamada actual está mal

    En tu `__main__` ponías:


    controller.disable_torque("follower_a")


    Eso antes no tenía sentido (no era “which”).
    Ahora correcto es:


    controller.disable_torque(roll="arm_roll", role="follower_a")


    o para deshabilitar todo:


    controller.disable_torque()


    Si me pegas **tu JSON real completo** (con todos los rolls) te lo dejo también con:

    * auto-detección de arm_rolls si no pasas ninguno,
    * validación de colisiones de IDs (evita duplicar el mismo servo en dos roles),
    * y un get_action_order() para que el orden de joints quede explícito en tu pipeline ACT.
'''