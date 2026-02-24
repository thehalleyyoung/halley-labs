# Real-world division-by-zero bugs from popular Python projects

# ── Bug 32: Empty list average (numpy/pandas pattern) ──
def compute_average(values):
    """Bug: division by zero when values is empty."""
    total = sum(values)
    count = len(values)
    return total / count  # BUG: count is 0 if values is empty

# ── Bug 33: Percentage calculation (BugsInPy tqdm-1) ──
def show_progress(done, total):
    """Bug: total could be 0 for empty tasks."""
    percent = (done / total) * 100  # BUG: total may be 0
    return f"{percent:.1f}%"

# ── Bug 34: Ratio with filter (scikit-learn pattern) ──
def class_weight(labels, target_class):
    """Bug: count of target_class may be 0."""
    total = len(labels)
    target_count = sum(1 for l in labels if l == target_class)
    weight = total / target_count  # BUG: target_count may be 0
    return weight

# ── Bug 35: Normalization (ML pattern) ──
def normalize_features(features):
    """Bug: max - min could be 0 for constant features."""
    min_val = min(features)
    max_val = max(features)
    span = max_val - min_val
    return [(f - min_val) / span for f in features]  # BUG: span may be 0

# ── Bug 36: FPS calculation (game/video pattern) ──
def calculate_fps(frames, elapsed_time):
    """Bug: elapsed_time could be 0 at start."""
    return frames / elapsed_time  # BUG: elapsed_time may be 0

# ── Bug 37: Throughput metric (web server pattern) ──
def throughput(requests_handled, duration_seconds):
    """Bug: duration could be sub-millisecond, rounded to 0."""
    return requests_handled / duration_seconds  # BUG: may be 0

# ── Bug 38: Guarded division (TRUE NEGATIVE) ──
def safe_average(values):
    """Not a bug: properly guarded."""
    if not values:
        return 0.0
    return sum(values) / len(values)

# ── Bug 39: Guarded with comparison (TRUE NEGATIVE) ──
def safe_ratio(a, b):
    """Not a bug: denominator checked."""
    if b == 0:
        return float('inf')
    return a / b

# ── Bug 40: Modulo by zero (common pattern) ──
def circular_index(items, step):
    """Bug: step could be 0."""
    return items[0 % step]  # BUG: step may be 0

# ── Bug 41: Integer division rounding (pandas pattern) ──
def chunk_data(data, num_chunks):
    """Bug: num_chunks could be 0."""
    chunk_size = len(data) // num_chunks  # BUG: num_chunks may be 0
    return [data[i:i+chunk_size] for i in range(0, len(data), max(chunk_size, 1))]

# ── Bug 42: Standard deviation (numpy-inspired) ──
def std_dev(values):
    """Bug: len(values) - 1 could be 0 for single-element list."""
    n = len(values)
    mean = sum(values) / n  # BUG #1: n could be 0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)  # BUG #2: n-1 could be 0
    return variance ** 0.5

# ── Bug 43: Rate calculation with time delta ──
def event_rate(events, start_time, end_time):
    """Bug: start and end could be same timestamp."""
    delta = end_time - start_time
    return len(events) / delta  # BUG: delta may be 0
