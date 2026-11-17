"""FK/IK minimal pour SO-101, sans placo : seulement numpy + scipy.
Angles en DEGRES (convention LeRobot 'new_calib'), positions en metres."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class SO101Kin:
    def __init__(self, urdf_path, target_link="gripper_frame_link"):
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], target_link
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        chain.reverse()
        self.steps = []  # (T_origin 4x4, axis ou None, nom)
        for j in chain:
            o = j.find("origin")
            T = np.eye(4)
            T[:3, :3] = R.from_euler("xyz", [float(v) for v in o.get("rpy").split()]).as_matrix()
            T[:3, 3] = [float(v) for v in o.get("xyz").split()]
            axis = None
            if j.get("type") == "revolute":
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            self.steps.append((T, axis, j.get("name")))
        lim = {j.get("name"): j.find("limit") for j in chain if j.get("type") == "revolute"}
        self.lo = np.array([np.degrees(float(lim[n].get("lower"))) for n in ARM])
        self.hi = np.array([np.degrees(float(lim[n].get("upper"))) for n in ARM])

    def fk(self, q_deg):
        q = dict(zip(ARM, np.radians(np.asarray(q_deg, float)[:5])))
        T = np.eye(4)
        for To, axis, name in self.steps:
            T = T @ To
            if axis is not None and name in q:
                Tr = np.eye(4)
                Tr[:3, :3] = R.from_rotvec(axis * q[name]).as_matrix()
                T = T @ Tr
        return T

    def ik(self, target_xyz, q_init_deg, reg=1e-3):
        """Position seule. Part de la pose actuelle q_init_deg (5 angles arm)."""
        q0 = np.clip(np.asarray(q_init_deg, float)[:5], self.lo, self.hi)
        tgt = np.asarray(target_xyz, float)

        def res(q):
            return np.concatenate([self.fk(q)[:3, 3] - tgt, reg * (q - q0) * np.radians(1)])

        sol = least_squares(res, q0, bounds=(self.lo, self.hi), xtol=1e-10, ftol=1e-10)
        err = np.linalg.norm(self.fk(sol.x)[:3, 3] - tgt)
        return sol.x, err

    def tilt_deg(self, q_deg):
        """Angle (deg) entre l'axe de la pince (z de gripper_frame_link) et la verticale vers le bas."""
        z = self.fk(q_deg)[:3, 2]
        return float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))

    def ik_down(self, target_xyz, q_init_deg, w_orient=0.3, reg=1e-3):
        """IK position + pince verticale (pointée vers le bas). Retourne (angles_deg, err_pos_m, tilt_deg).
        Teste plusieurs points de départ et garde la meilleure solution."""
        tgt = np.asarray(target_xyz, float)
        down = np.array([0.0, 0.0, -1.0])
        q_cur = np.clip(np.asarray(q_init_deg, float)[:5], self.lo, self.hi)
        seeds = [q_cur,
                 np.array([0.0, 15.0, 15.0, 60.0, 0.0]),
                 np.array([0.0, 40.0, -10.0, 60.0, 0.0])]
        best = None
        # wrist_roll n'influence pas l'axe de la pince : on le garde à sa valeur actuelle
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[4], hi[4] = q_cur[4] - 1e-6, q_cur[4] + 1e-6
        for q0 in seeds:
            q0 = np.clip(q0, lo, hi)
            q0[4] = q_cur[4]

            def res(q):
                T = self.fk(q)
                return np.concatenate([T[:3, 3] - tgt,
                                       w_orient * (T[:3, 2] - down),
                                       reg * (q - q_cur) * np.radians(1)])

            sol = least_squares(res, q0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
            err = float(np.linalg.norm(self.fk(sol.x)[:3, 3] - tgt))
            tilt = self.tilt_deg(sol.x)
            cost = err * 1000 + tilt          # 1 mm ~ 1 deg
            if best is None or cost < best[0]:
                best = (cost, sol.x, err, tilt)
        return best[1], best[2], best[3]
