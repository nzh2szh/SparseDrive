import pickle
import numpy as np
import os
import sys

P = os.path.join('work_dirs','sparsedrive_small_stage2_daq_rst','results.pkl')
if not os.path.exists(P):
    print('RESULTS_FILE_NOT_FOUND', P)
    sys.exit(2)

with open(P, 'rb') as f:
    data = pickle.load(f)


def try_array(x):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
            return arr
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            last = arr[:, -1]
            if np.all((last >= 0) & (last <= 1)):
                return last
    except Exception:
        return None
    return None


def extract_with_path(obj, path=''):
    """返回 (score, path) 列表，path 为键路径或属性路径"""
    out = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            newpath = f"{path}.{k}" if path else k
            # 如果键名里带 score 或 scores，优先尝试
            if 'score' in k.lower() or 'scores' in k.lower():
                arr = try_array(v)
                if arr is not None:
                    out.extend([(float(s), newpath) for s in arr.tolist()])
                    continue
            out.extend(extract_with_path(v, newpath))
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            newpath = f"{path}[{idx}]"
            out.extend(extract_with_path(v, newpath))
    else:
        for attr in ['scores_3d', 'scores', 'score', 'scores_2d']:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                arr = try_array(val)
                if arr is not None:
                    out.extend([(float(s), f"{path}.{attr}" if path else attr) for s in arr.tolist()])
        arr = try_array(obj)
        if arr is not None:
            out.extend([(float(s), path if path else 'root') for s in arr.tolist()])
    return out


items = extract_with_path(data)
if len(items) == 0:
    print('NO_SCORES_FOUND')
    sys.exit(3)

# 按关键字分类
map_keywords = ['map', 'lane', 'road', 'semantic_map']
obj_keywords = ['det', 'box', 'bbox', 'obj', 'object', 'pred', 'prediction', 'dets', 'boxes', 'track', 'target']

map_scores = []
obj_scores = []
other_scores = []
for s, path in items:
    lp = path.lower()
    if any(k in lp for k in map_keywords):
        map_scores.append(s)
    elif any(k in lp for k in obj_keywords):
        obj_scores.append(s)
    else:
        other_scores.append((s, path))

def summarize(arr):
    a = np.array(arr, dtype=float)
    return {'count': int(a.size), 'max': float(np.max(a)), 'min': float(np.min(a)), 'mean': float(np.mean(a))} if a.size>0 else None

sum_obj = summarize(obj_scores)
sum_map = summarize(map_scores)
sum_other = summarize([s for s, _ in other_scores]) if other_scores else None

def print_summary(name, summary, scores_array=None):
    if summary is None:
        print(f'{name}: NONE')
        return
    print(f'{name} - COUNT: {summary["count"]} MAX: {summary["max"]:.6f} MIN: {summary["min"]:.6f} MEAN: {summary["mean"]:.6f}')
    bins = np.linspace(0.0, 1.0, 6)
    hist, edges = np.histogram(np.array(scores_array, dtype=float), bins=bins)
    for i in range(len(hist)):
        left = edges[i]
        right = edges[i+1]
        label = f"{left:.1f}~{right:.1f}"
        print(f'  {label}: {int(hist[i])}')

print_summary('OBJECT_SCORES', sum_obj, obj_scores)
print_summary('MAP_SCORES', sum_map, map_scores)
print_summary('OTHER_SCORES', sum_other, [s for s, _ in other_scores] if other_scores else None)

outside = 0
if sum_obj is not None:
    outside += int(np.sum((np.array(obj_scores) < 0.0) | (np.array(obj_scores) > 1.0)))
if sum_map is not None:
    outside += int(np.sum((np.array(map_scores) < 0.0) | (np.array(map_scores) > 1.0)))
if sum_other is not None:
    outside += int(np.sum((np.array([s for s, _ in other_scores]) < 0.0) | (np.array([s for s, _ in other_scores]) > 1.0)))
if outside > 0:
    print(f'OUTSIDE_0_1: {outside}')
