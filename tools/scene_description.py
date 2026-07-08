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
        "event_ego_start_speed_thr": 0.8,
        "event_hard_brake_speed_drop": 0.8,
        "event_lane_change_lateral_shift_m": 2.5,
        "event_lane_change_min_forward_m": 4.0,
        "event_u_turn_heading_deg": 150.0,
        "event_follow_min_frames": 2,
        "event_cut_in_out_lateral_band_m": 1.5,
        "event_cut_in_enter_band_m": 1.2,
        "event_cut_out_exit_band_m": 2.0,
        "event_cut_min_lateral_shift_m": 0.8,
        "event_cut_min_track_frames": 3,
        "event_cut_confirm_frames": 2,
        "event_cut_in_out_front_distance_m": 25.0,
        "event_same_type_min_gap_sec": 2,
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


def build_event_description(events):
    if not events:
        return ""
    return " ".join(f"At t={item['t_sec']}s, {item['text']}." for item in events)


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
        return "stationary"
    if speed_mps < slow:
        return "slow"
    if speed_mps < fast:
        return "normal"
    return "fast"


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
    text = f"{agent['class_name']} {agent['position']} {agent['distance_m']:.1f}m"
    if speed_value is not None:
        speed_text = speed_phrase(speed_value, rules)
        if speed_text != "stationary":
            text += f", {speed_text}"
    if score_value is not None:
        text += f", s={score_value:.2f}"
    return text


def summarize_map_elements(map_elements, rules):
    if not map_elements:
        return rules.get("templates", {}).get("map_empty", "no significant map elements")
    parts = []
    for item in map_elements[:3]:
        text = f"{item['class_name']} {item['position']}"
        shape_text = item.get("shape_text")
        if shape_text:
            text += f", {shape_text}"
        parts.append(text)
    return ", ".join(parts)


def natural_position_phrase(position):
    replacements = {
        "front": "ahead",
        "rear": "behind",
        "left front": "ahead on the left",
        "right front": "ahead on the right",
        "left rear": "behind on the left",
        "right rear": "behind on the right",
        "near left side": "on the left",
        "near right side": "on the right",
        "near ego vehicle": "nearby",
        "center": "ahead",
    }
    return replacements.get(str(position), str(position))


def format_event_agent_text(agent, rules):
    speed_value = agent.get("speed_mps")
    speed_text = ""
    if speed_value is not None:
        speed = speed_phrase(speed_value, rules)
        if speed != "stationary":
            speed_text = f"{speed} "
    position_text = natural_position_phrase(agent.get("position"))
    return f"a {speed_text}{agent['class_name']} {agent['distance_m']:.1f}m {position_text}"


def format_event_map_text(item):
    position_text = natural_position_phrase(item.get("position"))
    shape_text = item.get("shape_text")
    if shape_text:
        shape_parts = str(shape_text).split("/")
        if len(shape_parts) == 3:
            shape_text = f"{shape_parts[0]}, {shape_parts[2]}"
    if shape_text:
        return f"a {shape_text} {item['class_name']} {position_text}"
    return f"a {item['class_name']} {position_text}"


def join_natural_phrases(items):
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def polyline_shape_text(points):
    points = to_numpy(points)
    if points is None or np.asarray(points).size == 0:
        return None
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    if points.shape[0] < 2 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    deltas = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.square(deltas[:, 0]) + np.square(deltas[:, 1]))
    length_m = float(np.sum(segment_lengths))
    direct_dx = safe_float(points[-1, 0] - points[0, 0])
    direct_dy = safe_float(points[-1, 1] - points[0, 1])
    direct_m = compute_distance(direct_dx, direct_dy)
    shape = "curved" if direct_m > 1e-3 and length_m / direct_m > 1.15 else "straight"

    if abs(direct_dx) >= abs(direct_dy) * 1.5:
        orientation = "longitudinal"
    elif abs(direct_dy) >= abs(direct_dx) * 1.5:
        orientation = "lateral"
    else:
        orientation = "diagonal"
    return f"{shape}/{orientation}/{length_m:.1f}m"


def build_event_context_text(agent_counts, salient_agents, map_elements, rules):
    segments = []
    if salient_agents:
        segments.extend(format_event_agent_text(agent, rules) for agent in salient_agents)
    elif agent_counts:
        segments.append(summarize_counts(agent_counts, rules, "det"))

    if map_elements:
        segments.extend(format_event_map_text(item) for item in map_elements[:2])

    if not segments:
        return ""
    return f"with {join_natural_phrases(segments)}"


def append_event_context(text, context_text):
    context_text = str(context_text or "").strip()
    if not context_text:
        return text
    return f"{text}, {context_text}"


def build_gt_event_context(info, rules):
    agent_counts, salient_agents = extract_agents_from_gt(info, rules)
    map_elements = extract_map_from_gt(info.get("map_annos"), rules)
    return build_event_context_text(agent_counts, salient_agents, map_elements, rules)


def build_pred_event_context(img_bbox, rules):
    agent_counts, salient_agents = extract_agents_from_pred(img_bbox, rules)
    map_elements = extract_map_from_pred(img_bbox, rules)
    return build_event_context_text(agent_counts, salient_agents, map_elements, rules)


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
        best_shape_text = None
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
                best_shape_text = polyline_shape_text(points)
        elements.append(
            {
                "raw_class_name": normalized_class_name,
                "class_name": alias_name(normalized_class_name, rules, "map"),
                "count": len(polylines),
                "position": best_position,
                "distance_m": best_distance,
                "shape_text": best_shape_text,
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
                "shape_text": polyline_shape_text(points),
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


def extract_planning_points(img_bbox):
    planning = to_numpy(img_bbox.get("final_planning"))
    if planning is None:
        planning = to_numpy(img_bbox.get("planning"))
    if planning is None:
        return None

    planning = np.asarray(planning)
    if planning.size == 0:
        return None

    while planning.ndim > 2:
        planning = planning[0]
    if planning.ndim == 1:
        if planning.size < 2:
            return None
        planning = planning.reshape(-1, 2)

    if planning.ndim != 2 or planning.shape[0] < 2:
        return None
    if planning.shape[1] < 2:
        return None
    if planning.shape[1] > 2:
        planning = planning[:, :2]
    return planning.astype(np.float32)


def planning_state_from_img_bbox(img_bbox):
    points = extract_planning_points(img_bbox)
    if points is None:
        return None

    head_count = min(len(points), 4)
    head_points = points[:head_count]
    step = np.diff(head_points, axis=0)
    step_norm = np.sqrt(np.square(step[:, 0]) + np.square(step[:, 1])) if len(step) > 0 else np.array([0.0])
    speed_proxy = float(np.mean(step_norm))

    end_point = points[-1]
    direction = end_point - points[0]
    heading_rad = None
    if float(np.hypot(direction[0], direction[1])) > 1e-3:
        heading_rad = float(math.atan2(direction[1], direction[0]))

    return {
        "speed_proxy": speed_proxy,
        "end_x": safe_float(end_point[0]),
        "end_y": safe_float(end_point[1]),
        "endpoint_dist": compute_distance(end_point[0], end_point[1]),
        "heading_rad": heading_rad,
    }


def normalize_angle_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def relative_t_sec(info, scene_start_timestamp):
    return int(math.floor(max(0.0, (info["timestamp"] - scene_start_timestamp) / 1e6)))


def quaternion_yaw(rotation):
    quat = to_numpy(rotation)
    if quat is None or np.asarray(quat).size < 4:
        return None
    quat = np.asarray(quat).reshape(-1)
    w, x_coord, y_coord, z_coord = [safe_float(value) for value in quat[:4]]
    siny_cosp = 2.0 * (w * z_coord + x_coord * y_coord)
    cosy_cosp = 1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord)
    return float(math.atan2(siny_cosp, cosy_cosp))


def gt_ego_motion_state(prev_info, curr_info):
    prev_translation = to_numpy(prev_info.get("ego2global_translation"))
    curr_translation = to_numpy(curr_info.get("ego2global_translation"))
    if prev_translation is None or curr_translation is None:
        return None
    prev_translation = np.asarray(prev_translation).reshape(-1)
    curr_translation = np.asarray(curr_translation).reshape(-1)
    if prev_translation.size < 2 or curr_translation.size < 2:
        return None

    dt_sec = (curr_info["timestamp"] - prev_info["timestamp"]) / 1e6
    if dt_sec <= 1e-3:
        return None

    dx_global = safe_float(curr_translation[0] - prev_translation[0])
    dy_global = safe_float(curr_translation[1] - prev_translation[1])
    distance_m = compute_distance(dx_global, dy_global)
    prev_yaw = quaternion_yaw(prev_info.get("ego2global_rotation"))
    curr_yaw = quaternion_yaw(curr_info.get("ego2global_rotation"))
    if prev_yaw is None:
        prev_yaw = math.atan2(dy_global, dx_global) if distance_m > 1e-3 else 0.0

    cos_yaw = math.cos(prev_yaw)
    sin_yaw = math.sin(prev_yaw)
    forward_m = dx_global * cos_yaw + dy_global * sin_yaw
    lateral_m = -dx_global * sin_yaw + dy_global * cos_yaw
    heading_delta_deg = None
    if curr_yaw is not None:
        heading_delta_deg = abs(normalize_angle_rad(curr_yaw - prev_yaw)) * 180.0 / math.pi

    return {
        "speed_mps": distance_m / dt_sec,
        "forward_m": forward_m,
        "lateral_m": lateral_m,
        "yaw_rad": curr_yaw,
        "heading_delta_deg": heading_delta_deg,
    }


def update_cut_track_state(prev_state, y_coord, rules):
    cut_min_lateral_shift = float(threshold(rules, "event_cut_min_lateral_shift_m", 0.8))
    min_track_frames = int(threshold(rules, "event_cut_min_track_frames", 3))
    confirm_frames = int(threshold(rules, "event_cut_confirm_frames", 2))

    if prev_state is None:
        return {
            "in_lane_center": cut_lane_center_state(y_coord, False, rules),
            "anchor_y": y_coord,
            "age": 1,
            "pending_state": None,
            "pending_count": 0,
        }, None

    age = int(prev_state.get("age", 0)) + 1
    prev_in_center = bool(prev_state.get("in_lane_center", False))
    prev_anchor_y = safe_float(prev_state.get("anchor_y", y_coord))
    candidate_center = cut_lane_center_state(y_coord, prev_in_center, rules)
    lateral_shift = abs(y_coord - prev_anchor_y)
    abs_delta = abs(y_coord) - abs(prev_anchor_y)

    new_state = dict(prev_state)
    new_state["age"] = age

    if candidate_center == prev_in_center:
        new_state["pending_state"] = None
        new_state["pending_count"] = 0
        return new_state, None

    if age < min_track_frames or lateral_shift < cut_min_lateral_shift:
        new_state["pending_state"] = None
        new_state["pending_count"] = 0
        return new_state, None

    if candidate_center and abs_delta > -cut_min_lateral_shift:
        return new_state, None
    if not candidate_center and abs_delta < cut_min_lateral_shift:
        return new_state, None

    pending_state = new_state.get("pending_state")
    pending_count = int(new_state.get("pending_count", 0))
    if pending_state == candidate_center:
        pending_count += 1
    else:
        pending_state = candidate_center
        pending_count = 1

    new_state["pending_state"] = pending_state
    new_state["pending_count"] = pending_count
    if pending_count < confirm_frames:
        return new_state, None

    new_state["in_lane_center"] = candidate_center
    new_state["anchor_y"] = y_coord
    new_state["pending_state"] = None
    new_state["pending_count"] = 0
    return new_state, "cut_in" if candidate_center else "cut_out"


def extract_lead_vehicle_distance_from_pred(img_bbox, rules):
    boxes = decode_pred_boxes(img_bbox.get("boxes_3d"))
    scores = to_numpy(img_bbox.get("scores_3d"))
    labels = to_numpy(img_bbox.get("labels_3d"))
    if scores is None or labels is None or boxes.size == 0:
        return None

    det_score_thr = float(threshold(rules, "det_score_thr", 0.25))
    vehicle_classes = set(rules.get("groups", {}).get("vehicle_classes", []))

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    lead_distance = None
    count = min(len(scores), len(labels), len(boxes))
    for index in range(count):
        score = safe_float(scores[index])
        if score < det_score_thr:
            continue
        label_index = int(labels[index])
        raw_class_name = DET_CLASS_NAMES[label_index] if 0 <= label_index < len(DET_CLASS_NAMES) else f"class_{label_index}"
        if raw_class_name not in vehicle_classes:
            continue
        box = boxes[index]
        x_coord = safe_float(box[0]) if len(box) > 0 else 0.0
        y_coord = safe_float(box[1]) if len(box) > 1 else 0.0
        if x_coord <= 0.0:
            continue
        distance_m = compute_distance(x_coord, y_coord)
        if lead_distance is None or distance_m < lead_distance:
            lead_distance = distance_m
    return lead_distance


def extract_vehicle_tracks_from_pred(img_bbox, rules):
    boxes = decode_pred_boxes(img_bbox.get("boxes_3d"))
    scores = to_numpy(img_bbox.get("scores_3d"))
    labels = to_numpy(img_bbox.get("labels_3d"))
    instance_ids = to_numpy(img_bbox.get("instance_ids"))
    if scores is None or labels is None or instance_ids is None or boxes.size == 0:
        return []

    det_score_thr = float(threshold(rules, "det_score_thr", 0.25))
    front_distance_limit = float(threshold(rules, "event_cut_in_out_front_distance_m", 25.0))
    vehicle_classes = set(rules.get("groups", {}).get("vehicle_classes", []))

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    instance_ids = np.asarray(instance_ids).reshape(-1)

    tracks = []
    count = min(len(scores), len(labels), len(instance_ids), len(boxes))
    for index in range(count):
        score = safe_float(scores[index])
        if score < det_score_thr:
            continue
        label_index = int(labels[index])
        raw_class_name = DET_CLASS_NAMES[label_index] if 0 <= label_index < len(DET_CLASS_NAMES) else f"class_{label_index}"
        if raw_class_name not in vehicle_classes:
            continue
        box = boxes[index]
        x_coord = safe_float(box[0]) if len(box) > 0 else 0.0
        y_coord = safe_float(box[1]) if len(box) > 1 else 0.0
        if x_coord <= 0.0:
            continue
        distance_m = compute_distance(x_coord, y_coord)
        if distance_m > front_distance_limit:
            continue
        tracks.append(
            {
                "instance_id": str(instance_ids[index]),
                "x_coord": x_coord,
                "y_coord": y_coord,
                "distance_m": distance_m,
            }
        )
    return tracks


def planning_state_from_gt_info(info):
    ego_trajs = to_numpy(info.get("gt_ego_fut_trajs"))
    if ego_trajs is None:
        return None
    ego_trajs = np.asarray(ego_trajs)
    if ego_trajs.size == 0:
        return None
    if ego_trajs.ndim == 1:
        if ego_trajs.size < 2:
            return None
        ego_trajs = ego_trajs.reshape(-1, 2)
    if ego_trajs.ndim != 2 or ego_trajs.shape[1] < 2:
        return None

    ego_masks = to_numpy(info.get("gt_ego_fut_masks"))
    if ego_masks is not None:
        ego_masks = np.asarray(ego_masks).reshape(-1)
        valid_len = int(np.sum(ego_masks > 0.5))
        if valid_len > 0:
            ego_trajs = ego_trajs[:valid_len]
    if len(ego_trajs) < 2:
        return None

    points = np.cumsum(ego_trajs[:, :2], axis=0)
    head_count = min(len(points), 4)
    head_points = points[:head_count]
    step = np.diff(head_points, axis=0)
    step_norm = np.sqrt(np.square(step[:, 0]) + np.square(step[:, 1])) if len(step) > 0 else np.array([0.0])
    speed_proxy = float(np.mean(step_norm))

    end_point = points[-1]
    direction = end_point - points[0]
    heading_rad = None
    if float(np.hypot(direction[0], direction[1])) > 1e-3:
        heading_rad = float(math.atan2(direction[1], direction[0]))

    return {
        "speed_proxy": speed_proxy,
        "end_x": safe_float(end_point[0]),
        "end_y": safe_float(end_point[1]),
        "endpoint_dist": compute_distance(end_point[0], end_point[1]),
        "heading_rad": heading_rad,
    }


def extract_lead_vehicle_distance_from_gt(info, rules):
    boxes = to_numpy(info.get("gt_boxes"))
    names = info.get("gt_names")
    if boxes is None or names is None:
        return None

    boxes = np.asarray(boxes)
    names = list(np.asarray(names).tolist())
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    vehicle_classes = set(rules.get("groups", {}).get("vehicle_classes", []))
    lead_distance = None
    count = min(len(boxes), len(names))
    for index in range(count):
        raw_class_name = str(names[index])
        if raw_class_name not in vehicle_classes:
            continue
        box = boxes[index]
        x_coord = safe_float(box[0]) if len(box) > 0 else 0.0
        y_coord = safe_float(box[1]) if len(box) > 1 else 0.0
        if x_coord <= 0.0:
            continue
        distance_m = compute_distance(x_coord, y_coord)
        if lead_distance is None or distance_m < lead_distance:
            lead_distance = distance_m
    return lead_distance


def extract_vehicle_tracks_from_gt(info, rules):
    boxes = to_numpy(info.get("gt_boxes"))
    names = info.get("gt_names")
    instance_inds = info.get("instance_inds")
    if boxes is None or names is None or instance_inds is None:
        return []

    boxes = np.asarray(boxes)
    names = list(np.asarray(names).tolist())
    instance_inds = list(np.asarray(instance_inds).tolist())
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    vehicle_classes = set(rules.get("groups", {}).get("vehicle_classes", []))
    front_distance_limit = float(threshold(rules, "event_cut_in_out_front_distance_m", 25.0))

    tracks = []
    count = min(len(boxes), len(names), len(instance_inds))
    for index in range(count):
        raw_class_name = str(names[index])
        if raw_class_name not in vehicle_classes:
            continue
        box = boxes[index]
        x_coord = safe_float(box[0]) if len(box) > 0 else 0.0
        y_coord = safe_float(box[1]) if len(box) > 1 else 0.0
        if x_coord <= 0.0:
            continue
        distance_m = compute_distance(x_coord, y_coord)
        if distance_m > front_distance_limit:
            continue
        tracks.append(
            {
                "instance_id": str(instance_inds[index]),
                "x_coord": x_coord,
                "y_coord": y_coord,
                "distance_m": distance_m,
            }
        )
    return tracks


def cut_lane_center_state(y_coord, prev_in_center, rules):
    legacy_band = float(threshold(rules, "event_cut_in_out_lateral_band_m", 1.5))
    enter_band = float(threshold(rules, "event_cut_in_enter_band_m", min(legacy_band, 1.2)))
    exit_band = float(threshold(rules, "event_cut_out_exit_band_m", max(legacy_band, enter_band)))
    if exit_band < enter_band:
        exit_band = enter_band

    abs_y = abs(safe_float(y_coord))
    if prev_in_center:
        return abs_y <= exit_band
    return abs_y <= enter_band


def detect_gt_scene_events(group, infos, rules):
    events = []
    last_type_t = {}
    last_cut_event_frame = {}
    prev_lane_state = {}
    prev_ego_speed = None
    ego_motion_history = []
    follow_run = 0
    follow_announced = False
    current_event_context_text = ""

    start_speed_thr = float(threshold(rules, "event_ego_start_speed_thr", 0.8))
    hard_brake_drop = float(threshold(rules, "event_hard_brake_speed_drop", 0.8))
    lane_change_lateral = float(threshold(rules, "event_lane_change_lateral_shift_m", 2.5))
    lane_change_forward = float(threshold(rules, "event_lane_change_min_forward_m", 4.0))
    u_turn_heading_deg = float(threshold(rules, "event_u_turn_heading_deg", 150.0))
    follow_distance = float(threshold(rules, "follow_vehicle_distance_m", 18.0))
    follow_min_frames = int(threshold(rules, "event_follow_min_frames", 2))
    same_type_min_gap = int(threshold(rules, "event_same_type_min_gap_sec", 2))

    def maybe_add_event(event_type, t_sec, text, dedupe_key=None, frame_index=None):
        key = dedupe_key or event_type
        if key in last_type_t and t_sec - last_type_t[key] < same_type_min_gap:
            return
        if frame_index is not None:
            prev_frame = last_cut_event_frame.get(key)
            if prev_frame is not None and frame_index - prev_frame < 2:
                return
            last_cut_event_frame[key] = frame_index
        last_type_t[key] = t_sec
        text = append_event_context(text, current_event_context_text)
        events.append({"t_sec": t_sec, "event_type": event_type, "text": text})

    for frame_index in range(group["start_index"], group["end_index"] + 1):
        info = infos[frame_index]
        t_sec = relative_t_sec(info, group["start_timestamp"])
        current_event_context_text = build_gt_event_context(info, rules)

        ego_motion = None
        if frame_index > group["start_index"]:
            ego_motion = gt_ego_motion_state(infos[frame_index - 1], info)
        if ego_motion is not None:
            ego_speed = ego_motion["speed_mps"]
            if prev_ego_speed is not None:
                if (
                    prev_ego_speed - ego_speed >= hard_brake_drop
                    and prev_ego_speed >= start_speed_thr
                    and ego_speed <= max(start_speed_thr, prev_ego_speed * 0.6)
                ):
                    maybe_add_event("ego_hard_brake", t_sec, "ego vehicle performs hard braking (approx)")

                stationary_speed = float(threshold(rules, "stationary_speed_mps", 0.3))
                if prev_ego_speed <= stationary_speed and ego_speed >= start_speed_thr:
                    maybe_add_event("ego_start", t_sec, "ego vehicle starts moving (approx)")

            prev_ego_speed = ego_speed
            ego_motion_history.append(ego_motion)
            ego_motion_history = ego_motion_history[-6:]

            lateral_sum = sum(item["lateral_m"] for item in ego_motion_history)
            forward_sum = sum(max(0.0, item["forward_m"]) for item in ego_motion_history)
            heading_sum = sum(item["heading_delta_deg"] or 0.0 for item in ego_motion_history)
            if (
                abs(lateral_sum) >= lane_change_lateral
                and forward_sum >= lane_change_forward
                and heading_sum <= 45.0
            ):
                direction = "to the left" if lateral_sum > 0.0 else "to the right"
                maybe_add_event("ego_lane_change", t_sec, f"ego vehicle changes lane {direction} (approx)")
                ego_motion_history = []

            if heading_sum >= u_turn_heading_deg and forward_sum >= lane_change_forward:
                maybe_add_event("ego_u_turn", t_sec, "ego vehicle performs a U-turn (approx)")
                ego_motion_history = []

        lead_distance = extract_lead_vehicle_distance_from_gt(info, rules)
        if lead_distance is not None and lead_distance <= follow_distance:
            follow_run += 1
            if follow_run >= follow_min_frames and not follow_announced:
                maybe_add_event("ego_follow", t_sec, "ego vehicle follows a lead vehicle (approx)")
                follow_announced = True
        else:
            follow_run = 0
            follow_announced = False

        tracks = extract_vehicle_tracks_from_gt(info, rules)
        for track in tracks:
            instance_id = track["instance_id"]
            y_coord = safe_float(track["y_coord"])
            prev_state = prev_lane_state.get(instance_id)
            new_state, cut_event = update_cut_track_state(prev_state, y_coord, rules)
            prev_lane_state[instance_id] = new_state
            if cut_event == "cut_in":
                maybe_add_event(
                    "other_cut_in",
                    t_sec,
                    f"a surrounding vehicle (id={instance_id}) cuts in (approx)",
                    dedupe_key=f"other_cut_in:{instance_id}",
                    frame_index=frame_index,
                )
            elif cut_event == "cut_out":
                maybe_add_event(
                    "other_cut_out",
                    t_sec,
                    f"a surrounding vehicle (id={instance_id}) cuts out (approx)",
                    dedupe_key=f"other_cut_out:{instance_id}",
                    frame_index=frame_index,
                )

    return sorted(events, key=lambda item: (item["t_sec"], item["event_type"]))


def detect_pred_scene_events(group, infos, results, rules):
    events = []
    last_type_t = {}
    last_cut_event_frame = {}
    prev_lane_state = {}
    prev_plan_state = None
    follow_run = 0
    follow_announced = False
    current_event_context_text = ""

    start_speed_thr = float(threshold(rules, "event_ego_start_speed_thr", 0.8))
    hard_brake_drop = float(threshold(rules, "event_hard_brake_speed_drop", 0.8))
    lane_change_lateral = float(threshold(rules, "event_lane_change_lateral_shift_m", 2.5))
    lane_change_forward = float(threshold(rules, "event_lane_change_min_forward_m", 4.0))
    u_turn_heading_deg = float(threshold(rules, "event_u_turn_heading_deg", 150.0))
    follow_distance = float(threshold(rules, "follow_vehicle_distance_m", 18.0))
    follow_min_frames = int(threshold(rules, "event_follow_min_frames", 2))
    same_type_min_gap = int(threshold(rules, "event_same_type_min_gap_sec", 2))

    def maybe_add_event(event_type, t_sec, text, dedupe_key=None, frame_index=None):
        key = dedupe_key or event_type
        if key in last_type_t and t_sec - last_type_t[key] < same_type_min_gap:
            return
        if frame_index is not None:
            prev_frame = last_cut_event_frame.get(key)
            if prev_frame is not None and frame_index - prev_frame < 2:
                return
            last_cut_event_frame[key] = frame_index
        last_type_t[key] = t_sec
        text = append_event_context(text, current_event_context_text)
        events.append({"t_sec": t_sec, "event_type": event_type, "text": text})

    for frame_index in range(group["start_index"], group["end_index"] + 1):
        info = infos[frame_index]
        img_bbox = results[frame_index].get("img_bbox", {})
        t_sec = relative_t_sec(info, group["start_timestamp"])
        current_event_context_text = build_pred_event_context(img_bbox, rules)

        plan_state = planning_state_from_img_bbox(img_bbox)
        if plan_state is not None and prev_plan_state is not None:
            if (
                prev_plan_state["speed_proxy"] - plan_state["speed_proxy"] >= hard_brake_drop
                and prev_plan_state["speed_proxy"] >= start_speed_thr
            ):
                maybe_add_event("ego_hard_brake", t_sec, "ego vehicle performs hard braking")

            if prev_plan_state["speed_proxy"] < start_speed_thr <= plan_state["speed_proxy"]:
                maybe_add_event("ego_start", t_sec, "ego vehicle starts moving")

            lateral_shift = abs(plan_state["end_y"] - prev_plan_state["end_y"])
            if lateral_shift >= lane_change_lateral and max(plan_state["end_x"], prev_plan_state["end_x"]) >= lane_change_forward:
                direction = "to the left" if plan_state["end_y"] > prev_plan_state["end_y"] else "to the right"
                maybe_add_event("ego_lane_change", t_sec, f"ego vehicle changes lane {direction}")

            prev_heading = prev_plan_state.get("heading_rad")
            curr_heading = plan_state.get("heading_rad")
            if prev_heading is not None and curr_heading is not None and plan_state["endpoint_dist"] >= lane_change_forward:
                delta_heading = abs(normalize_angle_rad(curr_heading - prev_heading)) * 180.0 / math.pi
                if delta_heading >= u_turn_heading_deg:
                    maybe_add_event("ego_u_turn", t_sec, "ego vehicle performs a U-turn")

        if plan_state is not None:
            prev_plan_state = plan_state

        lead_distance = extract_lead_vehicle_distance_from_pred(img_bbox, rules)
        if lead_distance is not None and lead_distance <= follow_distance:
            follow_run += 1
            if follow_run >= follow_min_frames and not follow_announced:
                maybe_add_event("ego_follow", t_sec, "ego vehicle follows a lead vehicle")
                follow_announced = True
        else:
            follow_run = 0
            follow_announced = False

        tracks = extract_vehicle_tracks_from_pred(img_bbox, rules)
        for track in tracks:
            instance_id = track["instance_id"]
            y_coord = safe_float(track["y_coord"])
            prev_state = prev_lane_state.get(instance_id)
            new_state, cut_event = update_cut_track_state(prev_state, y_coord, rules)
            prev_lane_state[instance_id] = new_state
            if cut_event == "cut_in":
                maybe_add_event(
                    "other_cut_in",
                    t_sec,
                    f"a surrounding vehicle (id={instance_id}) cuts in",
                    dedupe_key=f"other_cut_in:{instance_id}",
                    frame_index=frame_index,
                )
            elif cut_event == "cut_out":
                maybe_add_event(
                    "other_cut_out",
                    t_sec,
                    f"a surrounding vehicle (id={instance_id}) cuts out",
                    dedupe_key=f"other_cut_out:{instance_id}",
                    frame_index=frame_index,
                )

    return sorted(events, key=lambda item: (item["t_sec"], item["event_type"]))


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
    aggregate_counts = Counter()
    aggregate_map = Counter()
    scene_stats = init_scene_stats()
    errors = []

    for _, index in representative_indices(group, infos, rules):
        info = infos[index]
        try:
            agent_counts, salient_agents = extract_agents_from_gt(info, rules)
            map_elements = extract_map_from_gt(info.get("map_annos"), rules)
            for name, count in agent_counts.items():
                aggregate_counts[name] += count
            for item in map_elements:
                aggregate_map[item["raw_class_name"]] += item.get("count", 1)
            update_scene_stats(scene_stats, agent_counts, salient_agents, map_elements, rules)
        except Exception as exc:
            errors.append(f"sample_index={index}: {exc}")

    summary, scene_tags = build_scene_summary("gt", aggregate_counts, aggregate_map, 0, scene_stats, rules)
    events = detect_gt_scene_events(group, infos, rules)
    return {
        "scene_token": group["scene_token"],
        "start_timestamp": group["start_timestamp"],
        "start_index": group["start_index"],
        "end_index": group["end_index"],
        "summary": summary,
        "description": build_event_description(events),
        "scene_events": events,
        "scene_tags": scene_tags,
        "scene_stats": scene_stats,
        "errors": errors,
    }


def build_pred_scene_record(group, infos, results, rules):
    aggregate_counts = Counter()
    aggregate_map = Counter()
    instance_counter = Counter()
    scene_stats = init_scene_stats()
    errors = []

    for _, index in representative_indices(group, infos, rules):
        try:
            img_bbox = results[index].get("img_bbox", {})
            agent_counts, salient_agents = extract_agents_from_pred(img_bbox, rules)
            map_elements = extract_map_from_pred(img_bbox, rules)
            for name, count in agent_counts.items():
                aggregate_counts[name] += count
            for item in map_elements:
                aggregate_map[item["raw_class_name"]] += 1
            update_scene_stats(scene_stats, agent_counts, salient_agents, map_elements, rules)

            instance_ids = to_numpy(img_bbox.get("instance_ids"))
            if instance_ids is not None:
                for instance_id in np.asarray(instance_ids).reshape(-1).tolist():
                    instance_counter[str(instance_id)] += 1
        except Exception as exc:
            errors.append(f"sample_index={index}: {exc}")

    persistent_count = sum(
        1 for _, count in instance_counter.items()
        if count >= threshold(rules, "tracking_persistent_min_frames", 2)
    )
    summary, scene_tags = build_scene_summary("pred", aggregate_counts, aggregate_map, persistent_count, scene_stats, rules)
    events = detect_pred_scene_events(group, infos, results, rules)
    return {
        "scene_token": group["scene_token"],
        "start_timestamp": group["start_timestamp"],
        "start_index": group["start_index"],
        "end_index": group["end_index"],
        "summary": summary,
        "description": build_event_description(events),
        "scene_events": events,
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
        description_text = source_scene.get("description", "")
        item["description"] = append_text_sections(existing_description, description_text)
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