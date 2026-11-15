"""Apprend une correction de wrist_flex dépendant des autres articulations.

Mesures : pince physiquement verticale vers le BAS, moteurs désactivés.
Lancer avec OFFSETS = {} et CMD_OFFSETS = {} dans common.py.
Enregistre les données dans vertical_samples.csv et le modèle dans wrist_model.json.
ATTENTION : ce modèle empirique n'est valide que dans la zone échantillonnée.
"""
import csv
import json
from pathlib import Path
import numpy as np
from common import connect, read_joints

SAMPLES = Path(__file__).with_name('vertical_samples.csv')
MODEL = Path(__file__).with_name('wrist_model.json')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_roll', 'wrist_flex')
# Features des AUTRES angles. wrist_flex est la variable que l'on corrige.
FEATURES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_roll')
VERTICAL_SUM_DEG = 90.0  # hypothèse du modèle URDF ; l'offset absolu n'est pas identifiable sans référence
RIDGE = 1.0  # régularisation sur coefficients non constants, en degrés normalisés


def load_samples():
    if not SAMPLES.exists():
        return []
    with SAMPLES.open(newline='', encoding='utf-8') as f:
        return [{k: float(row[k]) for k in JOINTS} for row in csv.DictReader(f)]


def save_samples(samples):
    with SAMPLES.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=JOINTS)
        writer.writeheader()
        writer.writerows(samples)


def design(raw, center, scale):
    z = (raw - center) / scale
    # 1 + effets linéaires + carrés + interactions : régression polynomiale d'ordre 2.
    columns = [np.ones(len(z))]
    columns.extend(z[:, i] for i in range(4))
    columns.extend(z[:, i] ** 2 for i in range(4))
    columns.extend(z[:, i] * z[:, j] for i in range(4) for j in range(i + 1, 4))
    return np.column_stack(columns)


def fit(samples):
    raw = np.array([[s[k] for k in FEATURES] for s in samples], dtype=float)
    wrist = np.array([s['wrist_flex'] for s in samples], dtype=float)
    lift = raw[:, 1]
    elbow = raw[:, 2]
    # Chaque mesure fournit : wrist_corrige = 90 - lift - elbow.
    # Donc offset_wrist = 90 - lift - elbow - wrist_lu.
    y = VERTICAL_SUM_DEG - lift - elbow - wrist
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 1.0)
    A = design(raw, center, scale)
    penalty = np.diag([0.0] + [RIDGE] * (A.shape[1] - 1))
    beta = np.linalg.solve(A.T @ A + penalty, A.T @ y)
    prediction = A @ beta
    residual = y - prediction
    return {
        'features': list(FEATURES), 'center': center.tolist(), 'scale': scale.tolist(),
        'coefficients': beta.tolist(), 'min': raw.min(axis=0).tolist(),
        'max': raw.max(axis=0).tolist(), 'count': len(samples),
        'rmse_deg': float(np.sqrt(np.mean(residual ** 2))),
        'max_error_deg': float(np.max(np.abs(residual))),
        'vertical_sum_deg': VERTICAL_SUM_DEG,
    }, y, prediction


def predict_offset(q, model):
    """q: dict d'angles BRUTS, retourne l'offset à ajouter à wrist_flex."""
    x = np.array([[q[k] for k in model['features']]], dtype=float)
    A = design(x, np.asarray(model['center']), np.asarray(model['scale']))
    return float(A[0] @ np.asarray(model['coefficients']))


def main():
    samples = load_samples()
    print(f'{len(samples)} mesures déjà enregistrées dans {SAMPLES.name}.')
    print('Couple coupé : SOUTIENS LE BRAS. Pince réellement verticale vers le bas.')
    print('Varie largement épaule/coude, et si possible pan/roll, dans la zone utile.')
    print('Entrée = enregistrer ; q = calculer ; r = effacer les anciennes mesures.')
    robot = connect()
    robot.bus.disable_torque()
    running = True
    while running:
        cmd = input('> ').strip().lower()
        if cmd == 'q':
            running = False
        elif cmd == 'r':
            if input('Confirmer suppression des mesures ? (oui) : ').strip().lower() == 'oui':
                samples.clear()
                save_samples(samples)
        elif cmd == '':
            q = read_joints(robot, raw=True)
            sample = {k: float(q[k]) for k in JOINTS}
            samples.append(sample)
            save_samples(samples)
            y = 90.0 - sample['shoulder_lift'] - sample['elbow_flex'] - sample['wrist_flex']
            print(f"Mesure {len(samples)} : " + ', '.join(f'{k}={v:+.1f}' for k, v in sample.items())
                  + f' | offset local={y:+.2f}°')
        else:
            print('Commande inconnue : Entrée, q ou r.')
    print('Soutiens le bras : déconnexion, couple désactivé.')
    robot.disconnect()
    if len(samples) < 20:
        print(f'Pas assez de mesures ({len(samples)}/20). Le modèle quadratique a 15 paramètres : '
              'vise 30 à 50 poses variées avant de l’utiliser.')
        return
    model, actual, predicted = fit(samples)
    print(f"\nAjustement sur {len(samples)} poses : RMSE={model['rmse_deg']:.2f}°, "
          f"erreur max={model['max_error_deg']:.2f}°")
    print('Plages observées (degrés bruts) :')
    for k, lo, hi in zip(FEATURES, model['min'], model['max']):
        print(f'  {k}: {lo:+.1f} à {hi:+.1f}')
    if model['rmse_deg'] > 2.0 or model['max_error_deg'] > 5.0:
        print('ATTENTION : modèle imprécis. Ajoute des mesures variées, vérifie la verticalité et la calibration.')
        print('Modèle NON exporté.')
        return
    MODEL.write_text(json.dumps(model, indent=2), encoding='utf-8')
    print(f'Modèle enregistré dans {MODEL}. Ne pas remplacer OFFSETS par un nombre fixe.')
    print('IMPORTANT : il faut intégrer predict_offset() dans la lecture ET la commande du bras.')
    print('Le modèle n’est pas une correction de Z_TABLE et ne prouve pas la validité de l’URDF.')


if __name__ == '__main__':
    main()