import sys, time
import numpy as np
import common
from common import *
from utils import *
from urdf_ik import SO101Kin

# Orientation de la pince : angle (deg) = shoulder_pan + ROLL_SIGN * wrist_roll
# (wrist_roll = 0 -> la pince est orientée comme le bras ; l'angle passé en argument vaut alors shoulder_pan).
# Si la pince tourne du mauvais côté, passe ROLL_SIGN à -1.0. ROLL_OFFSET (deg) sert au réglage fin.
ROLL_SIGN = -1.0
ROLL_OFFSET = 10.0

# Sécurités assouplies pour les essais
IK_TOL = 0.025            # m : erreur IK tolérée (5 mm avant)

# Pince (échelle 0-100 de LeRobot : 0 = fermée, 100 = ouverte) -> À AJUSTER selon ton robot / l'objet
GRIPPER_OPEN = 50.0    # ouverture avant la saisie
GRIPPER_CLOSED = 0.0   # fermeture sur l'objet (mets 0 si tu veux fermer à fond)
GRIPPER_DUREE = 1.0    # durée (s) de l'ouverture / fermeture

if len(sys.argv) < 4:
    raise SystemExit("Usage : python pick_rota.py x y angle   (x, y en mètres, angle en degrés)")
x, y = float(sys.argv[1]), float(sys.argv[2])   # mètres
ANGLE = float(sys.argv[3])                      # degrés : orientation de la pince (0 = bras devant, pan = 0 et roll = 0)
kin = SO101Kin(URDF)
x_org, y_org, z_table = origin_xyz(kin)   # repère robot, déduit de la pose ORIGIN
robot = connect()


def set_gripper(value):
    move_to(robot, {"gripper": value}, duration_s=GRIPPER_DUREE)


try:
    if common.WAIT is not None:
        input("Entrée pour aller à la pose d'attente WAIT (Ctrl+C pour annuler)...")
        move_to(robot, common.WAIT)
        time.sleep(0.5)

    # La pince s'ouvre dès qu'on a atteint la position d'attente
    print("Ouverture de la pince...")
    set_gripper(GRIPPER_OPEN)
    time.sleep(0.3)

    q = read_joints(robot)
    q_now = np.array([q[j] for j in ARM])
    p_now = kin.fk(q_now)[:3, 3]
    target = [x_org + x, y_org + y, z_table + MARGE - Z_K * (x**2 + y**2)]   # repère robot
    print(f"DEBUG : x_org={x_org:.3f} y_org={y_org:.3f} z_table={z_table:.3f} | MARGE={MARGE} Z_K={Z_K} WRIST_K={WRIST_K}")
    print(f"point demandé (repère table) = ({x:.3f}, {y:.3f})  ->  repère robot = ({target[0]:.3f}, {target[1]:.3f})")

    # --- vérifications de cohérence (après une recalibration, les anciennes valeurs ne valent plus rien) ---
    for j, a, lo, hi in zip(ARM, q_now, kin.lo, kin.hi):
        if a < lo - 5 or a > hi + 5:
            print(f"ATTENTION : {j} lit {a:.1f} deg, hors des limites du modèle [{lo:.0f}, {hi:.0f}] : "
                  "calibration incohérente ou OFFSETS périmés.")
    if np.hypot(target[0], target[1]) > 0.5 or abs(target[2]) > 0.5:
        raise SystemExit(f"REFUSÉ : la cible est à {np.hypot(target[0], target[1])*1000:.0f} mm de la base. "
                         "Le bras ne dépasse pas ~400 mm : vérifie que ORIGIN et tes arguments "
                         "sont bien en MÈTRES.")

    sol, err, tilt = kin.ik_down(target, q_now)
    sol = np.array(sol, dtype=float)
    wrist_corr = WRIST_K * (x**2 + y**2)
    sol[ARM.index("wrist_flex")] += wrist_corr
    print(f"correction wrist_flex = {wrist_corr:+.2f} deg (WRIST_K = {WRIST_K})")

    # --- wrist_roll pour obtenir l'orientation demandée : ANGLE = pan + ROLL_SIGN * roll ---
    i_pan, i_roll = ARM.index("shoulder_pan"), ARM.index("wrist_roll")
    roll = ROLL_SIGN * (ANGLE - sol[i_pan]) + ROLL_OFFSET
    roll = (roll + 90.0) % 180.0 - 90.0           # la pince est symétrique : on ramène dans [-90, 90)
    candidats = [r for r in (roll, roll - 180.0, roll + 180.0) if kin.lo[i_roll] <= r <= kin.hi[i_roll]]
    if not candidats:
        raise SystemExit(f"REFUSÉ : wrist_roll = {roll:.1f} deg hors des limites [{kin.lo[i_roll]:.0f}, {kin.hi[i_roll]:.0f}].")
    sol[i_roll] = min(candidats, key=abs)
    print(f"orientation demandée {ANGLE:.1f} deg avec shoulder_pan = {sol[i_pan]:.1f} -> wrist_roll = {sol[i_roll]:.1f} deg")
    delta = sol - q_now
    print("pose actuelle (deg):", dict(zip(ARM, np.round(q_now, 1))), "xyz =", np.round(p_now, 3))
    print("pose cible    (deg):", dict(zip(ARM, np.round(sol, 1))), "xyz =", np.round(target, 3))
    print("variation     (deg):", dict(zip(ARM, np.round(delta, 1))), f"| erreur IK {err*1000:.2f} mm | inclinaison pince {tilt:.1f} deg")

    print(f"variation max : {np.max(np.abs(delta)):.1f} deg (limite {MAX_DELTA_DEG})")
    if err > IK_TOL:
        raise SystemExit(f"REFUSÉ : cible hors de portée (erreur IK {err*1000:.1f} mm > tolérance {IK_TOL*1000:.0f} mm).")
    if err > 0.005:
        print(f"ATTENTION : erreur IK {err*1000:.1f} mm (> 5 mm), la pince n'atteindra pas exactement la cible.")
    if target[2] < z_table - 0.01:
        print(f"ATTENTION : la cible est {1000*(z_table - target[2]):.0f} mm SOUS le plan de la table !")
    if tilt > MAX_TILT_DEG:
        raise SystemExit(f"REFUSÉ : pince inclinée de {tilt:.1f} deg (> {MAX_TILT_DEG}) : cible trop loin ou trop proche "
                         "pour garder la pince verticale. Distance à la base conseillée : environ 0,12 à 0,29 m.")
    if np.max(np.abs(delta)) > MAX_DELTA_DEG:
        raise SystemExit(f"REFUSÉ : un angle doit varier de plus de {MAX_DELTA_DEG} deg entre la pose actuelle "
                         "et la cible. Rapproche WAIT de la zone de travail (record_pose.py) "
                         "ou augmente MAX_DELTA_DEG.")

    # Rotation du poignet seule, depuis WAIT, avant d'amorcer la descente
    input("Entrée pour orienter le poignet (wrist_roll) (Ctrl+C pour annuler)...")
    move_to(robot, {"wrist_roll": sol[ARM.index("wrist_roll")]})
    time.sleep(0.5)

    input("Entrée pour descendre saisir l'objet (Ctrl+C pour annuler)...")
    move_to(robot, dict(zip(ARM, sol)), duration_s=DUREE)   # la pince reste ouverte pendant la descente
    time.sleep(0.5)

    descend_z(robot, kin, MARGE)
    print("Fermeture de la pince...")
    set_gripper(GRIPPER_CLOSED)
    time.sleep(0.5)

    # Retour à la pose d'attente en gardant l'objet
    if common.WAIT is not None:
        move_to(robot, {**common.WAIT, "gripper": GRIPPER_CLOSED})
    if common.POUBELLE is not None:
        move_to(robot, {**common.POUBELLE, "gripper": GRIPPER_CLOSED})
    input("Entrée pour relâcher l'objet et terminer (le bras reviendra au repos)...")
    set_gripper(GRIPPER_OPEN)
    time.sleep(0.5)
finally:
    safe_disconnect(robot)


