#!/usr/bin/env python3

import argparse
import copy
import json
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DET_CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

MAP_CLASS_NAMES = [
    "ped_crossing",
    "divider",
    "boundary",
]

DEFAULT_RULES_PATH = Path(__file__).with_name("scene_description_rules.json")

DEFAULT_RULES = {
    "sampling": {
        "timeline_stride_sec": 1,
        "time_mode": "floor",
    },
    "thresholds": {
        "det_score_thr": 0.25,
        "map_score_thr": 0.25,
        "max_salient_agents": 3,
        "near_longitudinal_m": 2.0,
        "lateral_side_m": 2.0,
        "stationary_speed_mps": 0.3,
        "slow_speed_mps": 2.0,
        "fast_speed_mps": 6.0,
        "dense_vehicle_count": 6,
        "dense_vru_count": 4,
        "ped_crossing_pedestrian_count": 2,
        "crossing_scene_min_hits": 1,
        "follow_vehicle_distance_m": 18.0,
        "follow_traffic_min_hits": 2,
        "follow_traffic_persistent_scene_count": 1,
        "close_boundary_distance_m": 6.0,
        "close_divider_distance_m": 6.0,
        "boundary_narrowing_min_hits": 2,
        "two_wheeler_front_distance_m": 15.0,
        "two_wheeler_weaving_lateral_m": 4.0,
        "two_wheeler_weaving_min_hits": 2,
        "tracking_persistent_min_frames": 2,
        "tracking_persistent_scene_count": 3,
    },
    "groups": {
        "vehicle_classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
        ],
        "vulnerable_classes": [
            "pedestrian",
            "bicycle",
            "motorcycle",
        ],
        "two_wheeler_classes": [
            "bicycle",
            "motorcycle",
        ],
    },
    "aliases": {
        "det": {
            "car": "car",
            "truck": "truck",
            "construction_vehicle": "construction vehicle",
            "bus": "bus",
            "trailer": "trailer",
            "barrier": "barrier",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
            "pedestrian": "pedestrian",
            "traffic_cone": "traffic cone",
        },
        "map": {
            "ped_crossing": "pedestrian crossing",
            "divider": "lane divider",
            "boundary": "road boundary",
        },
    },
    "templates": {
        "counts_empty": "no significant agents",
        "map_empty": "no significant map elements",
        "timeline_counts": "Primary agents are {counts}",
        "timeline_salient": "Notable agents include {salient}",
        "timeline_map": "Map context: {map_text}",
        "timeline_planning": "{planning}",
        "gt_summary": "This scene is mainly composed of {counts}",
        "pred_summary": "Model outputs suggest this scene mainly contains {counts}",
        "summary_map": "Common map elements include {map_text}",
        "summary_tracking": "There are {persistent_count} temporally persistent tracked agents",
        "summary_tags": "Scene characteristics can be summarized as {tags}",
    },
    "scene_tags": {
        "crossing_intersection": "intersection crossing",
        "ped_crossing_pedestrian": "active pedestrians near crossing",
        "ped_crossing_present": "pedestrian crossing ahead",
        "follow_traffic": "follow-traffic driving",
        "boundary_narrowing": "boundary narrowing",
        "two_wheeler_weaving": "two-wheeler weaving",
        "mixed_traffic": "mixed traffic with vehicles and two-wheelers",
        "vehicle_dense": "dense motor traffic",
        "vru_dense": "dense vulnerable road users",
        "lane_structure": "clear lane and boundary structure",
        "stable_tracking": "stable temporal tracking continuity",
        "urban_default": "typical urban road traffic",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene descriptions from SparseDrive predictions or NuScenes ground truth."
    )
    parser.add_argument("--infos", required=True, help="Path to infos pkl file.")
    parser.add_argument("--results", help="Path to SparseDrive results pkl file.")
    parser.add_argument(
        "--mode",
        choices=["pred", "gt", "both"],
        help="Generation mode. If omitted, it is inferred from outputs or available inputs.",
    )
    parser.add_argument("--pred-out", help="Output path for prediction-based scene descriptions.")
    parser.add_argument("--gt-out", help="Output path for ground-truth-based scene descriptions.")
    parser.add_argument(
        "--scene-json",
        help="Optional scene.json path to update by matching scene token. Disabled by default.",
    )
    parser.add_argument(
        "--scene-json-source",
        choices=["pred", "gt"],
        help="Select whether prediction or ground-truth descriptions are written back into --scene-json.",
    )
    parser.add_argument(
        "--rules-config",
        help="Optional JSON file that overrides default scene description rules.",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        help="Override predicted 3D object threshold in rules config.",
    )
    parser.add_argument(
        "--map-score-thr",
        type=float,
        help="Override predicted map vector threshold in rules config.",
    )
    parser.add_argument(
        "--max-salient-agents",
        type=int,
        help="Override maximum number of salient agents per timeline item.",
    )
    parser.add_argument(
        "--timeline-stride-sec",
        type=int,
        help="Override timeline sampling stride in seconds.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for JSON output.",
    )
    return parser.parse_args()


def ensure_parent_dir(path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_pickle(path_str):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pass

    with open(path_str, "rb") as file:
        return pickle.load(file)


def load_infos(path_str):
    payload = load_pickle(path_str)
    if isinstance(payload, dict) and isinstance(payload.get("infos"), list):
        return payload["infos"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported infos format: {type(payload).__name__}")


def load_results(path_str):
    payload = load_pickle(path_str)
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported results format: {type(payload).__name__}")
    return payload


def infer_generation_targets(args):
    if args.pred_out or args.gt_out:
        return args.pred_out is not None, args.gt_out is not None

    if args.mode == "pred":
        return True, False
    if args.mode == "gt":
        return False, True
    if args.mode == "both":
        return True, True

    if args.results:
        return True, True
    return False, True


def default_output_paths(want_pred, want_gt, pred_out, gt_out):
    pred_path = pred_out or ("scene_descriptions_pred.json" if want_pred else None)
    gt_path = gt_out or ("scene_descriptions_gt.json" if want_gt else None)
    return pred_path, gt_path


def to_numpy(data):
    if data is None:
        return None
    if hasattr(data, "tensor"):
        return to_numpy(data.tensor)

    try:
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ModuleNotFoundError:
        pass

    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    try:
        return np.asarray(data)
    except Exception:
        return None


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def deep_merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_scene_json_args(args, want_pred, want_gt):
    if not args.scene_json and not args.scene_json_source:
        return
    if not args.scene_json:
        raise ValueError("Writing back to scene.json requires --scene-json.")
    if not args.scene_json_source:
        raise ValueError("Writing back to scene.json requires --scene-json-source {pred|gt}.")
    if args.scene_json_source == "pred" and not want_pred:
        raise ValueError("--scene-json-source pred requires prediction descriptions to be generated.")
    if args.scene_json_source == "gt" and not want_gt:
        raise ValueError("--scene-json-source gt requires ground-truth descriptions to be generated.")


def to_camel_case(name):
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def camelize_keys(value):
    if isinstance(value, dict):
        return {to_camel_case(key): camelize_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [camelize_keys(item) for item in value]
    return value


def resolve_rules(args):
    rules = copy.deepcopy(DEFAULT_RULES)
    rules_path = None
    if DEFAULT_RULES_PATH.exists():
        rules = deep_merge_dict(rules, load_json(DEFAULT_RULES_PATH))
        rules_path = DEFAULT_RULES_PATH
    if args.rules_config:
        override_path = Path(args.rules_config)
        rules = deep_merge_dict(rules, load_json(override_path))
        rules_path = override_path

    thresholds = rules.setdefault("thresholds", {})
    sampling = rules.setdefault("sampling", {})
    if args.det_score_thr is not None:
        thresholds["det_score_thr"] = args.det_score_thr
    if args.map_score_thr is not None:
        thresholds["map_score_thr"] = args.map_score_thr
    if args.max_salient_agents is not None:
        thresholds["max_salient_agents"] = args.max_salient_agents
    if args.timeline_stride_sec is not None:
        sampling["timeline_stride_sec"] = args.timeline_stride_sec

    rules["runtime"] = {
        "rules_config": str(rules_path.resolve()) if rules_path else None,
    }
    return rules


def threshold(rules, name, default=None):
    return rules.get("thresholds", {}).get(name, default)


def alias_name(raw_name, rules, alias_group):
    aliases = rules.get("aliases", {}).get(alias_group, {})
    if raw_name in aliases:
        return str(aliases[raw_name])

    raw_name_str = str(raw_name)
    if raw_name_str in aliases:
        return str(aliases[raw_name_str])

    return raw_name_str


def normalize_map_class_name(raw_name):
    if isinstance(raw_name, (int, np.integer)):
        index = int(raw_name)
        if 0 <= index < len(MAP_CLASS_NAMES):
            return MAP_CLASS_NAMES[index]
        return f"map_{index}"

    raw_name_str = str(raw_name)
    if raw_name_str in MAP_CLASS_NAMES:
        return raw_name_str
    if raw_name_str.isdigit():
        index = int(raw_name_str)
        if 0 <= index < len(MAP_CLASS_NAMES):
            return MAP_CLASS_NAMES[index]
    return raw_name_str


def scene_groups(infos):
    if not infos:
        return []

    groups = []
    start_index = 0
    current_scene = infos[0]["scene_token"]
    for index in range(1, len(infos)):
        scene_token = infos[index]["scene_token"]
        if scene_token != current_scene:
            groups.append(
                {
                    "scene_token": current_scene,
                    "start_index": start_index,
                    "end_index": index - 1,
                    "start_timestamp": infos[start_index]["timestamp"],
                }
            )
            start_index = index
            current_scene = scene_token
    groups.append(
        {
            "scene_token": current_scene,
            "start_index": start_index,
            "end_index": len(infos) - 1,
            "start_timestamp": infos[start_index]["timestamp"],
        }
    )
    return groups


def merge_timeline_descriptions(items):
    merged = []
    for item in items:
        text = str(item.get("description", "")).strip()
        if not text:
            continue
        if text.endswith("."):
            text = text[:-1]
        merged.append(f"At t={item['t_sec']}s, {text}.")
    return " ".join(merged)


def representative_indices(group, infos, rules):
    per_bucket = {}
    stride_sec = max(1, int(rules.get("sampling", {}).get("timeline_stride_sec", 1)))
    time_mode = rules.get("sampling", {}).get("time_mode", "floor")
    start_timestamp = group["start_timestamp"]

    for index in range(group["start_index"], group["end_index"] + 1):
        rel_time = max(0.0, (infos[index]["timestamp"] - start_timestamp) / 1e6)
        if time_mode == "round":
            rel_sec = int(round(rel_time))
        else:
            rel_sec = int(math.floor(rel_time))
        bucket = (rel_sec // stride_sec) * stride_sec
        per_bucket.setdefault(bucket, index)
    return [(t_sec, index) for t_sec, index in sorted(per_bucket.items())]


def lateral_near_phrase(lateral):
    if lateral == "left":
        return "near left side"
    if lateral == "right":
        return "near right side"
    return "near ego vehicle"


def position_phrase(x_coord, y_coord, rules):
    near_longitudinal = threshold(rules, "near_longitudinal_m", 2.0)
    lateral_side = threshold(rules, "lateral_side_m", 2.0)

    longitudinal = "front" if x_coord >= 0.0 else "rear"
    if abs(x_coord) < near_longitudinal:
        longitudinal = "near"

    if y_coord > lateral_side:
        lateral = "left"
    elif y_coord < -lateral_side:
        lateral = "right"
    else:
        lateral = "center"

    if longitudinal == "near" and lateral == "center":
        return "near ego vehicle"
    if lateral == "center":
        return longitudinal
    if longitudinal == "near":
        return lateral_near_phrase(lateral)
    return f"{lateral} {longitudinal}"


def speed_phrase(speed_mps, rules):
    stationary = threshold(rules, "stationary_speed_mps", 0.3)
    slow = threshold(rules, "slow_speed_mps", 2.0)
    fast = threshold(rules, "fast_speed_mps", 6.0)
    if speed_mps < stationary:
        return "nearly stationary"
    if speed_mps < slow:
        return "moving slowly"
    if speed_mps < fast:
        return "moving at normal speed"
    return "moving fast"


def summarize_counts(counter_obj, rules, alias_group):
    if not counter_obj:
        return rules.get("templates", {}).get("counts_empty", "no significant agents")
    parts = []
    for raw_name, count in counter_obj.most_common(3):
        parts.append(f"{count} {alias_name(raw_name, rules, alias_group)}")
    return ", ".join(parts)


def summarize_map_counter(counter_obj, rules):
    if not counter_obj:
        return rules.get("templates", {}).get("map_empty", "no significant map elements")
    return ", ".join(alias_name(raw_name, rules, "map") for raw_name, _ in counter_obj.most_common(3))


def format_agent_text(agent, rules):
    speed_value = agent.get("speed_mps")
    score_value = agent.get("score")
    text = f"{agent['class_name']} at {agent['position']}, about {agent['distance_m']:.1f} m away"
    if speed_value is not None:
        text += f", {speed_phrase(speed_value, rules)}"
    if score_value is not None:
        text += f", score {score_value:.2f}"
    return text


def summarize_map_elements(map_elements, rules):
    if not map_elements:
        return rules.get("templates", {}).get("map_empty", "no significant map elements")
    return ", ".join(f"{item['class_name']} at {item['position']}" for item in map_elements[:3])


def build_timeline_description(agent_counts, salient_agents, map_elements, rules, planning_text=None):
    templates = rules.get("templates", {})
    segments = [
        templates.get("timeline_counts", "Primary agents are {counts}").format(
            counts=summarize_counts(agent_counts, rules, "det")
        )
    ]
    if salient_agents:
        salient_text = "; ".join(format_agent_text(agent, rules) for agent in salient_agents)
        segments.append(
            templates.get("timeline_salient", "Notable agents include {salient}").format(
                salient=salient_text
            )
        )
    if map_elements:
        segments.append(
            templates.get("timeline_map", "Map context: {map_text}").format(
                map_text=summarize_map_elements(map_elements, rules)
            )
        )
    if planning_text:
        segments.append(
            templates.get("timeline_planning", "{planning}").format(planning=planning_text)
        )
    return ". ".join(segment for segment in segments if segment) + "."


def compute_distance(x_coord, y_coord):
    return float(math.hypot(safe_float(x_coord), safe_float(y_coord)))


def init_scene_stats():
    return {
        "intersection_crossing_hit_count": 0,
        "follow_traffic_hit_count": 0,
        "close_boundary_hit_count": 0,
        "close_divider_hit_count": 0,
        "two_wheeler_weaving_hit_count": 0,
    }


def update_scene_stats(scene_stats, agent_counts, salient_agents, map_elements, rules):
    pedestrian_count = agent_counts.get("pedestrian", 0)
    ped_crossing_near = False
    boundary_near = False
    divider_near = False

    for item in map_elements:
        raw_class_name = item.get("raw_class_name")
        distance_m = safe_float(item.get("distance_m"), default=1e9)
        if raw_class_name == "ped_crossing":
            ped_crossing_near = True
        if raw_class_name == "boundary" and distance_m <= threshold(rules, "close_boundary_distance_m", 6.0):
            boundary_near = True
        if raw_class_name == "divider" and distance_m <= threshold(rules, "close_divider_distance_m", 6.0):
            divider_near = True

    if ped_crossing_near and pedestrian_count >= threshold(rules, "ped_crossing_pedestrian_count", 2):
        scene_stats["intersection_crossing_hit_count"] += 1
    if boundary_near:
        scene_stats["close_boundary_hit_count"] += 1
    if divider_near:
        scene_stats["close_divider_hit_count"] += 1

    vehicle_classes = set(rules.get("groups", {}).get("vehicle_classes", []))
    two_wheeler_classes = set(rules.get("groups", {}).get("two_wheeler_classes", []))
    follow_vehicle_distance = threshold(rules, "follow_vehicle_distance_m", 18.0)
    two_wheeler_front_distance = threshold(rules, "two_wheeler_front_distance_m", 15.0)
    weaving_lateral_limit = threshold(rules, "two_wheeler_weaving_lateral_m", 4.0)

    for agent in salient_agents:
        raw_class_name = agent.get("raw_class_name")
        x_coord = safe_float(agent.get("x_coord"))
        distance_m = safe_float(agent.get("distance_m"), default=1e9)
        if raw_class_name in vehicle_classes and x_coord > 0.0 and distance_m <= follow_vehicle_distance:
            scene_stats["follow_traffic_hit_count"] += 1
            break

    for agent in salient_agents:
        raw_class_name = agent.get("raw_class_name")
        x_coord = safe_float(agent.get("x_coord"))
        y_coord = safe_float(agent.get("y_coord"))
        distance_m = safe_float(agent.get("distance_m"), default=1e9)
        if (
            raw_class_name in two_wheeler_classes
            and x_coord > 0.0
            and distance_m <= two_wheeler_front_distance
            and abs(y_coord) <= weaving_lateral_limit
        ):
            scene_stats["two_wheeler_weaving_hit_count"] += 1
            break


def extract_map_from_gt(map_annos, rules):
    elements = []
    if not isinstance(map_annos, dict):
        return elements

    for raw_class_name, polylines in map_annos.items():
        normalized_class_name = normalize_map_class_name(raw_class_name)
        if not polylines:
            continue
        best_distance = None
        best_position = "near ego vehicle"
        for polyline in polylines:
            points = to_numpy(polyline)
            if points is None or points.size == 0:
                continue
            points = np.asarray(points)
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            distances = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
            min_index = int(np.argmin(distances))
            distance = float(distances[min_index])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_position = position_phrase(points[min_index, 0], points[min_index, 1], rules)
        elements.append(
            {
                "raw_class_name": normalized_class_name,
                "class_name": alias_name(normalized_class_name, rules, "map"),
                "count": len(polylines),
                "position": best_position,
                "distance_m": best_distance,
            }
        )
    return sorted(elements, key=lambda item: (item["distance_m"] is None, item["distance_m"] or 0.0))[:3]


def extract_agents_from_gt(info, rules):
    boxes = to_numpy(info.get("gt_boxes"))
    names = info.get("gt_names")
    velocities = to_numpy(info.get("gt_velocity"))
    max_salient_agents = int(threshold(rules, "max_salient_agents", 3))
    if boxes is None or names is None:
        return Counter(), []

    names = list(np.asarray(names).tolist())
    if velocities is None:
        velocities = np.zeros((len(names), 2), dtype=np.float32)

    counts = Counter()
    agents = []
    for index, raw_class_name in enumerate(names):
        counts[raw_class_name] += 1
        box = boxes[index]
        velocity = velocities[index] if index < len(velocities) else [0.0, 0.0]
        x_coord = safe_float(box[0])
        y_coord = safe_float(box[1])
        speed_mps = float(math.hypot(safe_float(velocity[0]), safe_float(velocity[1])))
        agents.append(
            {
                "raw_class_name": raw_class_name,
                "class_name": alias_name(raw_class_name, rules, "det"),
                "x_coord": x_coord,
                "y_coord": y_coord,
                "position": position_phrase(x_coord, y_coord, rules),
                "distance_m": compute_distance(x_coord, y_coord),
                "speed_mps": speed_mps,
            }
        )
    agents.sort(key=lambda item: item["distance_m"])
    return counts, agents[:max_salient_agents]


def decode_pred_boxes(boxes_3d):
    boxes = to_numpy(boxes_3d)
    if boxes is None:
        return np.zeros((0, 0), dtype=np.float32)
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    return boxes


def extract_agents_from_pred(img_bbox, rules):
    boxes = decode_pred_boxes(img_bbox.get("boxes_3d"))
    scores = to_numpy(img_bbox.get("scores_3d"))
    labels = to_numpy(img_bbox.get("labels_3d"))
    max_salient_agents = int(threshold(rules, "max_salient_agents", 3))
    det_score_thr = float(threshold(rules, "det_score_thr", 0.25))
    if scores is None or labels is None or boxes.size == 0:
        return Counter(), []

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    counts = Counter()
    agents = []
    for index in range(min(len(scores), len(labels), len(boxes))):
        score = safe_float(scores[index])
        if score < det_score_thr:
            continue
        label_index = int(labels[index])
        raw_class_name = DET_CLASS_NAMES[label_index] if 0 <= label_index < len(DET_CLASS_NAMES) else f"class_{label_index}"
        box = boxes[index]
        x_coord = safe_float(box[0]) if len(box) > 0 else 0.0
        y_coord = safe_float(box[1]) if len(box) > 1 else 0.0
        velocity_x = safe_float(box[7]) if len(box) > 7 else 0.0
        velocity_y = safe_float(box[8]) if len(box) > 8 else 0.0
        counts[raw_class_name] += 1
        agents.append(
            {
                "raw_class_name": raw_class_name,
                "class_name": alias_name(raw_class_name, rules, "det"),
                "x_coord": x_coord,
                "y_coord": y_coord,
                "position": position_phrase(x_coord, y_coord, rules),
                "distance_m": compute_distance(x_coord, y_coord),
                "speed_mps": float(math.hypot(velocity_x, velocity_y)),
                "score": score,
            }
        )
    agents.sort(key=lambda item: item["distance_m"])
    return counts, agents[:max_salient_agents]


def extract_map_from_pred(img_bbox, rules):
    vectors = img_bbox.get("vectors")
    scores = to_numpy(img_bbox.get("scores"))
    labels = to_numpy(img_bbox.get("labels"))
    map_score_thr = float(threshold(rules, "map_score_thr", 0.25))
    if vectors is None or scores is None or labels is None:
        return []

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    elements = []
    count = min(len(scores), len(labels), len(vectors))
    for index in range(count):
        score = safe_float(scores[index])
        if score < map_score_thr:
            continue
        label_index = int(labels[index])
        raw_class_name = MAP_CLASS_NAMES[label_index] if 0 <= label_index < len(MAP_CLASS_NAMES) else f"map_{label_index}"
        points = to_numpy(vectors[index])
        if points is None or np.asarray(points).size == 0:
            continue
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        distances = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        min_index = int(np.argmin(distances))
        elements.append(
            {
                "raw_class_name": raw_class_name,
                "class_name": alias_name(raw_class_name, rules, "map"),
                "position": position_phrase(points[min_index, 0], points[min_index, 1], rules),
                "distance_m": float(distances[min_index]),
                "score": score,
            }
        )
    return sorted(elements, key=lambda item: item["distance_m"])[:3]


def extract_planning_text(img_bbox, rules):
    planning = to_numpy(img_bbox.get("final_planning"))
    if planning is None:
        planning = to_numpy(img_bbox.get("planning"))
    if planning is None or np.asarray(planning).size == 0:
        return None
    planning = np.asarray(planning)
    endpoint = planning[-1] if planning.ndim >= 2 else planning.reshape(-1)
    if len(endpoint) < 2:
        return None
    end_x = safe_float(endpoint[0])
    end_y = safe_float(endpoint[1])
    return f"Planned ego endpoint is around {position_phrase(end_x, end_y, rules)}, about {compute_distance(end_x, end_y):.1f} m away"


def derive_scene_tags(agent_counter, map_counter, persistent_count, scene_stats, rules):
    tags = []
    scene_tags = rules.get("scene_tags", {})
    groups = rules.get("groups", {})

    vehicle_count = sum(agent_counter.get(name, 0) for name in groups.get("vehicle_classes", []))
    vru_count = sum(agent_counter.get(name, 0) for name in groups.get("vulnerable_classes", []))
    two_wheeler_count = sum(agent_counter.get(name, 0) for name in groups.get("two_wheeler_classes", []))

    if scene_stats.get("intersection_crossing_hit_count", 0) >= threshold(rules, "crossing_scene_min_hits", 1):
        tags.append(scene_tags.get("crossing_intersection", "intersection crossing"))
    elif map_counter.get("ped_crossing", 0) > 0 and agent_counter.get("pedestrian", 0) >= threshold(rules, "ped_crossing_pedestrian_count", 2):
        tags.append(scene_tags.get("ped_crossing_pedestrian", "active pedestrians near crossing"))
    elif map_counter.get("ped_crossing", 0) > 0:
        tags.append(scene_tags.get("ped_crossing_present", "pedestrian crossing ahead"))

    if (
        scene_stats.get("follow_traffic_hit_count", 0) >= threshold(rules, "follow_traffic_min_hits", 2)
        and persistent_count >= threshold(rules, "follow_traffic_persistent_scene_count", 1)
    ):
        tags.append(scene_tags.get("follow_traffic", "follow-traffic driving"))

    if scene_stats.get("two_wheeler_weaving_hit_count", 0) >= threshold(rules, "two_wheeler_weaving_min_hits", 2):
        tags.append(scene_tags.get("two_wheeler_weaving", "two-wheeler weaving"))
    elif vehicle_count >= threshold(rules, "dense_vehicle_count", 6) and two_wheeler_count > 0:
        tags.append(scene_tags.get("mixed_traffic", "mixed traffic with vehicles and two-wheelers"))
    elif vehicle_count >= threshold(rules, "dense_vehicle_count", 6):
        tags.append(scene_tags.get("vehicle_dense", "dense motor traffic"))

    if vru_count >= threshold(rules, "dense_vru_count", 4):
        tags.append(scene_tags.get("vru_dense", "dense vulnerable road users"))

    if (
        scene_stats.get("close_boundary_hit_count", 0) >= threshold(rules, "boundary_narrowing_min_hits", 2)
        and scene_stats.get("close_divider_hit_count", 0) >= threshold(rules, "boundary_narrowing_min_hits", 2)
    ):
        tags.append(scene_tags.get("boundary_narrowing", "boundary narrowing"))
    elif map_counter.get("divider", 0) > 0 and map_counter.get("boundary", 0) > 0:
        tags.append(scene_tags.get("lane_structure", "clear lane and boundary structure"))

    if persistent_count >= threshold(rules, "tracking_persistent_scene_count", 3):
        tags.append(scene_tags.get("stable_tracking", "stable temporal tracking continuity"))

    if not tags and agent_counter:
        tags.append(scene_tags.get("urban_default", "typical urban road traffic"))

    unique_tags = []
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags


def build_scene_summary(source_type, aggregate_counts, aggregate_map, persistent_count, scene_stats, rules):
    templates = rules.get("templates", {})
    if source_type == "pred":
        base = templates.get("pred_summary", "Model outputs suggest this scene mainly contains {counts}")
    else:
        base = templates.get("gt_summary", "This scene is mainly composed of {counts}")

    summary_parts = [base.format(counts=summarize_counts(aggregate_counts, rules, "det"))]
    if aggregate_map:
        summary_parts.append(
            templates.get("summary_map", "Common map elements include {map_text}").format(
                map_text=summarize_map_counter(aggregate_map, rules)
            )
        )
    if source_type == "pred" and persistent_count > 0:
        summary_parts.append(
            templates.get("summary_tracking", "There are {persistent_count} temporally persistent tracked agents").format(
                persistent_count=persistent_count
            )
        )

    tags = derive_scene_tags(aggregate_counts, aggregate_map, persistent_count, scene_stats, rules)
    if tags:
        summary_parts.append(
            templates.get("summary_tags", "Scene characteristics can be summarized as {tags}").format(tags=", ".join(tags))
        )
    return ", ".join(summary_parts) + ".", tags


def build_gt_scene_record(group, infos, rules):
    timeline_descriptions = []
    aggregate_counts = Counter()
    aggregate_map = Counter()
    scene_stats = init_scene_stats()
    errors = []

    for t_sec, index in representative_indices(group, infos, rules):
        info = infos[index]
        try:
            agent_counts, salient_agents = extract_agents_from_gt(info, rules)
            map_elements = extract_map_from_gt(info.get("map_annos"), rules)
            for name, count in agent_counts.items():
                aggregate_counts[name] += count
            for item in map_elements:
                aggregate_map[item["raw_class_name"]] += item.get("count", 1)
            update_scene_stats(scene_stats, agent_counts, salient_agents, map_elements, rules)
            timeline_descriptions.append(
                {
                    "t_sec": t_sec,
                    "description": build_timeline_description(agent_counts, salient_agents, map_elements, rules),
                }
            )
        except Exception as exc:
            errors.append(f"sample_index={index}: {exc}")

    summary, scene_tags = build_scene_summary("gt", aggregate_counts, aggregate_map, 0, scene_stats, rules)
    return {
        "scene_token": group["scene_token"],
        "start_timestamp": group["start_timestamp"],
        "start_index": group["start_index"],
        "end_index": group["end_index"],
        "summary": summary,
        "description": merge_timeline_descriptions(timeline_descriptions),
        "scene_tags": scene_tags,
        "scene_stats": scene_stats,
        "errors": errors,
    }


def build_pred_scene_record(group, infos, results, rules):
    timeline_descriptions = []
    aggregate_counts = Counter()
    aggregate_map = Counter()
    instance_counter = Counter()
    scene_stats = init_scene_stats()
    errors = []

    for t_sec, index in representative_indices(group, infos, rules):
        try:
            img_bbox = results[index].get("img_bbox", {})
            agent_counts, salient_agents = extract_agents_from_pred(img_bbox, rules)
            map_elements = extract_map_from_pred(img_bbox, rules)
            planning_text = extract_planning_text(img_bbox, rules)
            for name, count in agent_counts.items():
                aggregate_counts[name] += count
            for item in map_elements:
                aggregate_map[item["raw_class_name"]] += 1
            update_scene_stats(scene_stats, agent_counts, salient_agents, map_elements, rules)

            instance_ids = to_numpy(img_bbox.get("instance_ids"))
            if instance_ids is not None:
                for instance_id in np.asarray(instance_ids).reshape(-1).tolist():
                    instance_counter[str(instance_id)] += 1

            timeline_descriptions.append(
                {
                    "t_sec": t_sec,
                    "description": build_timeline_description(
                        agent_counts,
                        salient_agents,
                        map_elements,
                        rules,
                        planning_text=planning_text,
                    ),
                }
            )
        except Exception as exc:
            errors.append(f"sample_index={index}: {exc}")

    persistent_count = sum(
        1 for _, count in instance_counter.items()
        if count >= threshold(rules, "tracking_persistent_min_frames", 2)
    )
    summary, scene_tags = build_scene_summary("pred", aggregate_counts, aggregate_map, persistent_count, scene_stats, rules)
    return {
        "scene_token": group["scene_token"],
        "start_timestamp": group["start_timestamp"],
        "start_index": group["start_index"],
        "end_index": group["end_index"],
        "summary": summary,
        "description": merge_timeline_descriptions(timeline_descriptions),
        "scene_tags": scene_tags,
        "scene_stats": scene_stats,
        "tracking_summary": {
            "persistent_instance_count": persistent_count,
        },
        "errors": errors,
    }


def build_output_payload(source_type, infos_path, results_path, scenes, args, rules):
    payload = {
        "source_type": source_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scenes),
        "input_files": {
            "infos": str(Path(infos_path).resolve()),
        },
        "config": {
            "det_score_thr": threshold(rules, "det_score_thr", 0.25),
            "map_score_thr": threshold(rules, "map_score_thr", 0.25),
            "max_salient_agents": threshold(rules, "max_salient_agents", 3),
            "timeline_stride_sec": rules.get("sampling", {}).get("timeline_stride_sec", 1),
            "rules_config": rules.get("runtime", {}).get("rules_config"),
        },
        "scenes": scenes,
    }
    if results_path:
        payload["input_files"]["results"] = str(Path(results_path).resolve())
    return camelize_keys(payload)


def validate_gt_infos(infos):
    required = ["scene_token", "timestamp", "gt_boxes", "gt_names", "gt_velocity", "map_annos"]
    if not infos:
        raise ValueError("Infos file is empty.")
    missing = [key for key in required if key not in infos[0]]
    if missing:
        raise ValueError(f"Infos file is missing required GT fields: {', '.join(missing)}")


def validate_pred_inputs(results, infos):
    if results is None:
        raise ValueError("Prediction mode requires --results.")
    if len(results) != len(infos):
        raise ValueError(
            f"Length mismatch between results and infos: {len(results)} vs {len(infos)}"
        )


def write_json(path_str, payload, indent):
    path = ensure_parent_dir(path_str)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=indent)
        file.write("\n")


def scene_entries_from_payload(payload):
    return payload.get("scenes", []) if isinstance(payload, dict) else []


def resolve_scene_json_items(scene_payload):
    if isinstance(scene_payload, list):
        return scene_payload
    if isinstance(scene_payload, dict) and isinstance(scene_payload.get("scenes"), list):
        return scene_payload["scenes"]
    raise ValueError("Unsupported scene.json format. Expected a top-level list or an object with a 'scenes' list.")


def scene_token_of(item):
    if not isinstance(item, dict):
        return None
    return item.get("sceneToken") or item.get("scene_token") or item.get("token")


def append_text_sections(*sections):
    normalized = []
    for section in sections:
        text = str(section or "").strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text = f"{text}."
        normalized.append(text)
    return " ".join(normalized)


def update_scene_json(scene_json_path, source_payload, indent):
    scene_json = load_json(scene_json_path)
    scene_items = resolve_scene_json_items(scene_json)
    source_by_token = {
        scene.get("sceneToken"): scene
        for scene in scene_entries_from_payload(source_payload)
        if isinstance(scene, dict) and scene.get("sceneToken")
    }

    for item in scene_items:
        token = scene_token_of(item)
        if token is None or token not in source_by_token:
            continue
        source_scene = source_by_token[token]
        existing_description = item.get("description", "")
        summary_text = source_scene.get("summary", "")
        description_text = source_scene.get("description", "")
        item["description"] = append_text_sections(existing_description, summary_text, description_text)
        if "summary" in item:
            del item["summary"]

    write_json(scene_json_path, scene_json, indent)


def main():
    args = parse_args()
    rules = resolve_rules(args)
    want_pred, want_gt = infer_generation_targets(args)
    validate_scene_json_args(args, want_pred, want_gt)
    pred_out, gt_out = default_output_paths(want_pred, want_gt, args.pred_out, args.gt_out)

    infos = load_infos(args.infos)
    groups = scene_groups(infos)
    results = None

    if want_pred:
        if not args.results:
            raise ValueError("Prediction description generation requires --results.")
        results = load_results(args.results)
        validate_pred_inputs(results, infos)

    if want_gt:
        validate_gt_infos(infos)

    if want_pred:
        pred_scenes = [build_pred_scene_record(group, infos, results, rules) for group in groups]
        pred_payload = build_output_payload("pred", args.infos, args.results, pred_scenes, args, rules)
        write_json(pred_out, pred_payload, args.indent)
        print(f"Wrote prediction descriptions to {pred_out}")
        if args.scene_json_source == "pred":
            update_scene_json(args.scene_json, pred_payload, args.indent)
            print(f"Updated scene json from prediction descriptions: {args.scene_json}")

    if want_gt:
        gt_scenes = [build_gt_scene_record(group, infos, rules) for group in groups]
        gt_payload = build_output_payload("gt", args.infos, None, gt_scenes, args, rules)
        write_json(gt_out, gt_payload, args.indent)
        print(f"Wrote ground-truth descriptions to {gt_out}")
        if args.scene_json_source == "gt":
            update_scene_json(args.scene_json, gt_payload, args.indent)
            print(f"Updated scene json from ground-truth descriptions: {args.scene_json}")


if __name__ == "__main__":
    main()