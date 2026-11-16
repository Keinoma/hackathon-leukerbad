"""Exercise Windows sharing violations without requiring Windows or robot hardware."""
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from leukerbad.detection.base import Detection
from leukerbad.workspace import export_target, write_json


def test_atomic_replace_retries_windows_permission_errors(monkeypatch, tmp_path):
    import leukerbad.workspace as workspace
    path = tmp_path / 'state.json'
    path.write_text('{"old":true}')
    replace = workspace.os.replace
    attempts = []
    def blocked_then_available(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert json.loads(path.read_text()) == {'old': True}
            raise PermissionError(5, 'Access denied')
        replace(source, destination)
    monkeypatch.setattr(workspace.os, 'replace', blocked_then_available)
    monkeypatch.setattr(workspace.time, 'sleep', lambda delay: None)
    write_json(path, {'new': True})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {'new': True}
    assert len(list(tmp_path.iterdir())) == 1


def test_persistent_lock_preserves_old_file_and_removes_temporary(monkeypatch, tmp_path):
    import leukerbad.workspace as workspace
    path = tmp_path / 'state.json'
    path.write_text('{"old":true}')
    def denied(*args):
        raise PermissionError(5, 'Access denied')
    monkeypatch.setattr(workspace.os, 'replace', denied)
    monkeypatch.setattr(workspace.time, 'sleep', lambda delay: None)
    with pytest.raises(PermissionError):
        write_json(path, {'new': True})
    assert json.loads(path.read_text()) == {'old': True}
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('confidence,ready', [(.59, False), (.60, True), (.9, True)])
def test_export_enforces_pick_threshold(tmp_path, confidence, ready):
    calibration = {'image_size': [500,500], 'points_px': [[10,10],[410,10],[410,410],[10,410]],
                   'width_mm':385, 'height_mm':280, 'origin_px':[210,210]}
    polygon = cv2.boxPoints(((200,200),(80,20),30))
    payload = export_target(tmp_path/'state.json', calibration, (500,500),
                            [Detection('vape',confidence,(0,0,0,0))], {0:polygon})
    assert (payload['target'] is not None) is ready


def robot_reader():
    spec=importlib.util.spec_from_file_location('robot_reader_sharing',Path(__file__).parents[1]/'main.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('confidence', [.59, None])
def test_robot_rejects_low_or_missing_confidence(tmp_path, confidence):
    module=robot_reader()
    target={'x_mm':1,'y_mm':2,'theta_deg':3}
    if confidence is not None:
        target['confidence']=confidence
    path=tmp_path/'state.json'
    path.write_text(json.dumps({'state':'ready','timestamp':time.time(),'target':target}))
    with pytest.raises(module.NoTarget):
        module.load_target(path)


def test_reader_treats_access_denied_as_retryable(monkeypatch,tmp_path):
    module=robot_reader()
    path=tmp_path/'state.json'
    path.write_text('{}')
    def denied(*args,**kwargs):
        raise PermissionError(5,'Access denied')
    monkeypatch.setattr('builtins.open',denied)
    with pytest.raises(module.NoTarget):
        module.load_target(path)


def test_camera_survives_locked_json_including_exit(monkeypatch, tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace

    from leukerbad.cli import run_camera
    data = {'image_size':[500,500], 'points_px':[[10,10],[410,10],[410,410],[10,410]],
            'width_mm':385,'height_mm':280,'origin_px':[210,210]}
    path = tmp_path/'workspace.json'
    path.write_text(json.dumps(data))
    image = np.zeros((500,500,3),np.uint8)
    released=[]
    camera=SimpleNamespace(isOpened=lambda:True,release=lambda:released.append(True))
    frames=SimpleNamespace(read=lambda:(True,image),close=lambda:None)
    detector=SimpleNamespace(analyse=lambda *a,**k:(image.copy(),[],{}))
    monkeypatch.setattr(cv2,'VideoCapture',lambda *a:camera)
    monkeypatch.setattr('leukerbad.cli.LatestFrames',lambda camera:frames)
    for name in ('namedWindow','setMouseCallback','imshow','destroyAllWindows'):
        monkeypatch.setattr(cv2,name,lambda *a:None)
    monkeypatch.setattr(cv2,'getWindowProperty',lambda *a:1)
    keys=iter([-1,ord('q')])
    monkeypatch.setattr(cv2,'waitKey',lambda *a:next(keys))
    attempts=[]
    def denied(*args,**kwargs):
        attempts.append(1)
        raise PermissionError(5,'Access denied')
    monkeypatch.setattr('leukerbad.workspace.export_target',denied)
    run_camera(Namespace(camera=0,calibration=path,json_out=tmp_path/'state.json',lens_file=None),detector)
    assert len(attempts)==3 and released==[True]
