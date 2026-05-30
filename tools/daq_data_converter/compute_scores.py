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

scores = []

def try_array(x):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
            return arr
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            # heuristic: last column might be score if values in [0,1]
            last = arr[:, -1]
            if np.all((last >= 0) & (last <= 1)):
                return last
    except Exception:
        return None
    return None


def extract(obj):
    out = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'score' in k.lower() or 'scores' in k.lower():
                arr = try_array(v)
                if arr is not None:
                    out.extend(arr.tolist())
                    continue
            out.extend(extract(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(extract(v))
    else:
        # try attribute access
        for attr in ['scores_3d','scores','score','scores_2d']:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                arr = try_array(val)
                if arr is not None:
                    out.extend(arr.tolist())
        # try array-like
        arr = try_array(obj)
        if arr is not None:
            out.extend(arr.tolist())
    return out

scores = extract(data)

if len(scores) == 0:
    print('NO_SCORES_FOUND')
    sys.exit(3)

scores = np.array(scores, dtype=float)
count = scores.size
mx = float(np.max(scores))
mn = float(np.min(scores))
mean = float(np.mean(scores))
print(f'COUNT: {count}')
print(f'MAX: {mx:.6f}')
print(f'MIN: {mn:.6f}')
print(f'MEAN: {mean:.6f}')
# 统计 0.2 步长的区间：[0.0,0.2),[0.2,0.4),...,[0.8,1.0]
bins = np.linspace(0.0, 1.0, 6)
hist, edges = np.histogram(scores, bins=bins)
for i in range(len(hist)):
    left = edges[i]
    right = edges[i+1]
    # 对最后一个区间包含右端点
    if i == len(hist)-1:
        label = f"{left:.1f}~{right:.1f}"
    else:
        label = f"{left:.1f}~{right:.1f}"
    print(f'{label}: {int(hist[i])}')
# 统计超出 [0,1] 的数量
outside = np.sum((scores < 0.0) | (scores > 1.0))
if outside > 0:
    print(f'OUTSIDE_0_1: {int(outside)}')
