#!/usr/bin/env python3
"""Convert daq_data_rst samples into a NuScenes-style pickle.

Usage:
    python scripts/daq_data_converter/daq_data_converter.py \
            -i /abs/path/to/data/daq_data_rst \
            -o /abs/path/to/output \
    -c 45
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


FRAME_DIR_RE = re.compile(r"^\d+_\d{3}$")
FRAME_INFO_PREPROC_NAME = "frame_info_preproc.json"
CAN_BUS_JSON_NAME = "can_bus.json"
OUTPUT_PKL_NAME = "daq_data_infos_infe.pkl"
CAMERA_NAME_MAP = {
    "CAM_FRONT": "FrontCam02_preproc",
    "CAM_FRONT_RIGHT": "SideFrontCam02_preproc",
    "CAM_FRONT_LEFT": "SideFrontCam01_preproc",
    "CAM_BACK": "RearCam01_preproc",
    "CAM_BACK_LEFT": "SideRearCam01_preproc",
    "CAM_BACK_RIGHT": "SideRearCam02_preproc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert daq_data_rst into a NuScenes-style pickle")
    parser.add_argument("-i", "--root-path", dest="root_path", required=True, help="源数据根目录绝对路径")
    parser.add_argument("-o", "--out-dir", dest="out_dir", required=True, help="结果输出绝对路径")
    parser.add_argument("-c", "--steering-thresh-deg", dest="steering_thresh_deg", type=float, default=45.0, help="方向盘转角阈值角度，默认 45，单位为度")
    return parser.parse_args()


def load_relaxed_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = strip_json_comments(text)
    text = text.translate(str.maketrans({"：": ":", "，": ",", "；": ";"}))
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def strip_json_comments(text: str) -> str:
    stripped_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        comment_index = None
        for index in range(len(line) - 1):
            char = line[index]
            next_char = line[index + 1]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string and char == "/" and next_char == "/":
                comment_index = index
                break
        stripped_lines.append(line[:comment_index] if comment_index is not None else line)
    return "\n".join(stripped_lines)


def list_frame_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and FRAME_DIR_RE.match(path.name)),
        key=lambda path: path.name,
    )


def list_scene_dirs(root: Path) -> list[Path]:
    frame_dirs = list_frame_dirs(root)
    if frame_dirs:
        return [root]
    return sorted((path for path in root.iterdir() if path.is_dir() and list_frame_dirs(path)), key=lambda path: path.name)


def find_can_bus_path(scene_dir: Path, root_dir: Path) -> Path | None:
    candidate = scene_dir / CAN_BUS_JSON_NAME
    if candidate.is_file():
        return candidate
    candidate = root_dir / CAN_BUS_JSON_NAME
    if candidate.is_file():
        return candidate
    return None


def find_single_file(directory: Path, suffix: str) -> Path | None:
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix))
    if not matches:
        return None
    return matches[0]


def find_frame_info_path(sample_dir: Path) -> Path:
    path = sample_dir / FRAME_INFO_PREPROC_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少 frame_info_preproc.json: {path}")
    return path


def find_camera_image(sample_dir: Path, camera_name: str) -> Path:
    matches = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file()
        and (
            path.name.endswith(f"_{camera_name}.jpeg")
            or path.name.endswith(f"_{camera_name}.jpg")
            or path.name.endswith(f"_{camera_name}.png")
        )
    )
    if matches:
        return matches[0]

    matches = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and f"_{camera_name}." in path.name
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"缺少相机图像: {sample_dir} / {camera_name}")


def find_lidar_extrinsic_path(scene_dir: Path, root_dir: Path) -> Path:
    candidate_paths = [
        scene_dir / "xxtrinsic" / "lidar_to_vehicle_extrinsic.yaml",
        root_dir / "xxtrinsic" / "lidar_to_vehicle_extrinsic.yaml",
        scene_dir / "xxtrinsic" / "lidar_to_vehicle_extrinsic.yml",
        root_dir / "xxtrinsic" / "lidar_to_vehicle_extrinsic.yml",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"缺少 lidar_to_vehicle_extrinsic.yaml: {scene_dir}")


def load_opencv_matrix(path: Path, key: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    pattern = rf"(?ms)^{re.escape(key)}:\s*!!opencv-matrix\s*.*?^\s*data:\s*\[(.*?)\]\s*$"
    match = re.search(pattern, text)
    if match is None:
        raise KeyError(f"在 {path} 中未找到矩阵 {key}")

    raw_values = [value for value in re.split(r"[\s,]+", match.group(1).strip()) if value]
    values = np.array([float(value) for value in raw_values], dtype=np.float64)
    return values


def load_lidar_to_vehicle_matrix(path: Path) -> np.ndarray:
    values = load_opencv_matrix(path, "T_v_l0")
    if values.size != 16:
        raise ValueError(f"T_v_l0 期望 16 个元素，实际得到 {values.size}: {path}")
    return values.reshape(4, 4)


def matrix_to_translation_rotation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform shape must be 4x4, got {transform.shape}")
    translation = transform[:3, 3].astype(np.float64)
    rotation = matrix_to_quaternion_wxyz(transform[:3, :3])
    return translation, rotation


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation matrix shape must be 3x3, got {rotation.shape}")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag = np.diag(rotation)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
        elif axis == 1:
            scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
            qw = (rotation[1, 0] - rotation[0, 1]) / scale

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm > 0.0:
        quaternion /= norm
    return quaternion


def extract_sensor_to_ego_transform(camera: dict[str, Any]) -> np.ndarray:
    ego_to_sensor = np.asarray(camera.get("T_c_b_static", []), dtype=np.float64)
    if ego_to_sensor.shape != (4, 4):
        raise ValueError(f"T_c_b_static shape must be 4x4, got {ego_to_sensor.shape}")
    return np.linalg.inv(ego_to_sensor)


def extract_camera_entry(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsic = np.asarray(camera.get("camera_matrix", []), dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera_matrix shape must be 3x3, got {intrinsic.shape}")

    transform = extract_sensor_to_ego_transform(camera)
    sensor2ego_translation = transform[:3, 3].astype(np.float64)
    sensor2ego_rotation = matrix_to_quaternion_wxyz(transform[:3, :3])
    return intrinsic, sensor2ego_translation, sensor2ego_rotation


def normalize_steering_value(entry: dict[str, Any]) -> float:
    value = float(entry.get("value", 0.0))
    unit = str(entry.get("unit", "rad")).lower()
    if unit in {"deg", "degree", "degrees"}:
        return math.radians(value)
    return value


def nearest_can_value(can_series: list[dict[str, Any]], frame_timestamp_s: float) -> float:
    if not can_series:
        return 0.0

    nearest_entry = min(
        can_series,
        key=lambda entry: abs(float(entry.get("timestamp", 0.0)) - frame_timestamp_s),
    )
    return normalize_steering_value(nearest_entry)


def steering_to_cmd(steering_rad: float, steering_thresh_deg: float) -> np.ndarray:
    thresh_rad = math.radians(steering_thresh_deg)
    if steering_rad >= thresh_rad:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if steering_rad <= -thresh_rad:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def build_cam_info(
    sample_dir: Path,
    camera_name: str,
    camera: dict[str, Any],
    lidar_to_vehicle: np.ndarray,
) -> dict[str, Any]:
    intrinsic, sensor2ego_translation, sensor2ego_rotation = extract_camera_entry(camera)
    sensor2ego = extract_sensor_to_ego_transform(camera)

    lidar_to_vehicle = np.asarray(lidar_to_vehicle, dtype=np.float64)
    if lidar_to_vehicle.shape != (4, 4):
        raise ValueError(f"lidar_to_vehicle shape must be 4x4, got {lidar_to_vehicle.shape}")

    sensor2lidar = np.linalg.inv(lidar_to_vehicle) @ sensor2ego
    sensor2lidar_translation = sensor2lidar[:3, 3].astype(np.float64)
    sensor2lidar_rotation = sensor2lidar[:3, :3].astype(np.float64)
    image_path = find_camera_image(sample_dir, camera_name)
    return {
        "data_path": str(image_path.resolve()),
        "cam_intrinsic": intrinsic,
        "sensor2ego_translation": sensor2ego_translation.tolist(),
        "sensor2ego_rotation": sensor2ego_rotation.tolist(),
        "sensor2lidar_rotation": sensor2lidar_rotation.tolist(),
        "sensor2lidar_translation": sensor2lidar_translation.tolist(),
    }


def build_info(
    sample_dir: Path,
    frame_info: dict[str, Any],
    steering_rad: float,
    sample_token: str,
    prev_token: str | None,
    next_token: str | None,
    scene_token: str,
    steering_thresh_deg: float,
    lidar_to_vehicle: np.ndarray,
) -> dict[str, Any]:
    cameras = frame_info.get("camera_calibration", {}).get("cameras", [])
    camera_by_name = {str(camera.get("name", "")): camera for camera in cameras}

    front_camera = camera_by_name.get("FrontCam02_preproc")
    if front_camera is None:
        raise KeyError(f"缺少 FrontCam02_preproc 相机信息: {sample_dir}")

    front_timestamp_ns = int(front_camera.get("timestamp", 0))
    timestamp_us = int(round(front_timestamp_ns / 1000.0))

    pose = front_camera.get("pose", {})
    ego_translation = np.asarray(pose.get("position", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    ego_rotation_wxyz = np.asarray(pose.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(4)
    ego_rotation_wxyz = ego_rotation_wxyz / max(float(np.linalg.norm(ego_rotation_wxyz)), 1.0)

    vc = np.asarray(front_camera.get("vc", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    ego_status = np.array(
        [vc[0], vc[1], vc[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, steering_rad],
        dtype=np.float32,
    )

    lidar_translation, lidar_rotation_wxyz = matrix_to_translation_rotation(lidar_to_vehicle)

    lidar_path = find_single_file(sample_dir, "MainLidar01_demotion.pcd")

    cams: dict[str, dict[str, Any]] = {}
    for camera_key, camera_name in CAMERA_NAME_MAP.items():
        camera = camera_by_name.get(camera_name)
        if camera is None:
            raise KeyError(f"缺少相机 {camera_name}: {sample_dir}")
        cams[camera_key] = build_cam_info(sample_dir, camera_name, camera, lidar_to_vehicle)

    return {
        "token": sample_token,
        "timestamp": timestamp_us,
        "cams": cams,
        "ego2global_translation": ego_translation.tolist(),
        "ego2global_rotation": ego_rotation_wxyz.tolist(),
        "ego_status": ego_status,
        "prev": prev_token,
        "next": next_token,
        "scene_token": scene_token,
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
        "gt_names": np.array([], dtype=object),
        "gt_velocity": np.zeros((0, 2), dtype=np.float32),
        "gt_trajectory": [],
        "map_location": "",
        "lidar_path": str(lidar_path.resolve()) if lidar_path is not None else "",
        "sweeps": [],
        "lidar2ego_translation": lidar_translation.tolist(),
        "lidar2ego_rotation": lidar_rotation_wxyz.tolist(),
        "map_annos": {},
        "num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "valid_flag": np.zeros((0,), dtype=bool),
        "gt_ego_fut_trajs": np.zeros((0, 2), dtype=np.float32),
        "gt_ego_fut_masks": np.zeros((0,), dtype=np.float32),
        "gt_ego_fut_cmd": steering_to_cmd(steering_rad, steering_thresh_deg),
    }


def iter_sample_dirs(scene_dir: Path) -> list[Path]:
    return sorted(
        (path for path in scene_dir.iterdir() if path.is_dir() and FRAME_DIR_RE.match(path.name)),
        key=lambda path: path.name,
    )


def build_infos(root_dir: Path, steering_thresh_deg: float) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []

    scene_dirs = list_scene_dirs(root_dir)
    if not scene_dirs:
        raise FileNotFoundError(f"未找到可转换的场景目录: {root_dir}")

    for scene_dir in scene_dirs:
        sample_dirs = iter_sample_dirs(scene_dir)
        if not sample_dirs:
            continue

        lidar_extrinsic_path = find_lidar_extrinsic_path(scene_dir, root_dir)
        lidar_to_vehicle = load_lidar_to_vehicle_matrix(lidar_extrinsic_path)

        can_bus_path = find_can_bus_path(scene_dir, root_dir)
        can_series = []
        if can_bus_path is not None:
            can_bus = load_relaxed_json(can_bus_path)
            can_series = list(can_bus.get("steeringAngle", []))

        scene_token = sample_dirs[0].name if scene_dir == root_dir else scene_dir.name
        prev_token: str | None = None

        for index, sample_dir in enumerate(sample_dirs):
            sample_token = sample_dir.relative_to(root_dir).as_posix().replace("/", "_")
            next_token = (
                sample_dirs[index + 1].relative_to(root_dir).as_posix().replace("/", "_")
                if index + 1 < len(sample_dirs)
                else None
            )
            frame_info = load_relaxed_json(find_frame_info_path(sample_dir))
            front_camera = next(
                camera
                for camera in frame_info.get("camera_calibration", {}).get("cameras", [])
                if str(camera.get("name", "")) == "FrontCam02_preproc"
            )
            front_timestamp_ns = int(front_camera.get("timestamp", 0))
            frame_timestamp_s = front_timestamp_ns / 1_000_000_000.0
            steering_rad = nearest_can_value(can_series, frame_timestamp_s)
            if scene_dir == root_dir:
                scene_token = sample_dirs[0].name
            else:
                scene_token = scene_dir.relative_to(root_dir).as_posix().replace("/", "_")

            infos.append(
                build_info(
                    sample_dir=sample_dir,
                    frame_info=frame_info,
                    steering_rad=steering_rad,
                    sample_token=sample_token,
                    prev_token=prev_token,
                    next_token=next_token,
                    scene_token=scene_token,
                    steering_thresh_deg=steering_thresh_deg,
                    lidar_to_vehicle=lidar_to_vehicle,
                )
            )
            prev_token = sample_token

    return infos


def dump_pkl(out_dir: Path, infos: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_PKL_NAME
    payload = {
        "infos": infos,
        "metadata": {
            "version": "v1.0-trainval",
        },
    }
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not root_dir.is_dir():
        print(f"输入目录不存在或不是目录: {root_dir}", file=sys.stderr)
        return 2

    infos = build_infos(root_dir, args.steering_thresh_deg)
    if not infos:
        print(f"未生成任何 info，检查输入目录: {root_dir}", file=sys.stderr)
        return 1

    out_path = dump_pkl(out_dir, infos)
    print(f"转换完成，已写入 {len(infos)} 条样本到: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
