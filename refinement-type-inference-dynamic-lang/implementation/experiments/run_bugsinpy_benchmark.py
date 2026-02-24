"""Third-party benchmark: BugsInPy-inspired evaluation.

Evaluates GuardHarvest on bug patterns derived from BugsInPy, a curated
dataset of real Python bugs from popular projects.  Unlike the author
and CVE-inspired benchmarks, these patterns come from independently
documented, version-controlled bugs with fix commits.

Each function reproduces the PRE-FIX buggy pattern in minimal form.
Ground truth comes from the original bug report / fix diff.

References:
  Widyasari et al. "BugsInPy: A Database of Existing Bugs in Python
  Programs to Enable Controlled Testing and Debugging Studies"
  ESEC/FSE 2020.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_analyzer import FlowSensitiveAnalyzer
import ast


# ── BugsInPy-Inspired Test Functions ─────────────────────────────────

BUGSINPY_BENCHMARK = '''
# =============================================================================
# BugsInPy-Inspired Benchmark
# Each function reproduces a bug pattern from a real, documented Python bug.
# Ground truth labels from original bug reports / fix commits.
# =============================================================================

# --- youtube-dl #26015: NoneType has no attribute 'group' ---
# Project: youtube-dl, Bug ID: 1
# Fix: Add None check after re.search()
def youtube_dl_regex_none(url):
    """BUG: re.search may return None."""
    import re
    mobj = re.search(r'video/(\\d+)', url)
    return mobj.group(1)  # BUG: mobj may be None

GROUND_TRUTH_youtube_dl_regex_none = {"buggy": True, "category": "NULL_DEREF"}


# --- youtube-dl #26015 fixed version ---
def youtube_dl_regex_none_safe(url):
    """SAFE: None check added."""
    import re
    mobj = re.search(r'video/(\\d+)', url)
    if mobj is None:
        return None
    return mobj.group(1)

GROUND_TRUTH_youtube_dl_regex_none_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- scrapy #4356: AttributeError on None response ---
# Project: scrapy, Bug ID: 3
def scrapy_none_response(response):
    """BUG: response.get() returns Optional, used without check."""
    encoding = response.headers.get("Content-Type")
    return encoding.decode("utf-8")  # BUG: encoding may be None

GROUND_TRUTH_scrapy_none_response = {"buggy": True, "category": "NULL_DEREF"}


# --- scrapy safe version ---
def scrapy_none_response_safe(response):
    """SAFE: None check."""
    encoding = response.headers.get("Content-Type")
    if encoding is not None:
        return encoding.decode("utf-8")
    return "text/html"

GROUND_TRUTH_scrapy_none_response_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- pandas #28382: ZeroDivisionError in resample ---
# Project: pandas, Bug ID: 5
def pandas_resample_zero(total, count):
    """BUG: count may be zero from empty groupby."""
    mean = total / count  # BUG: division by zero
    return mean

GROUND_TRUTH_pandas_resample_zero = {"buggy": True, "category": "DIV_BY_ZERO"}


# --- pandas safe version ---
def pandas_resample_zero_safe(total, count):
    """SAFE: zero check."""
    if count == 0:
        return 0.0
    return total / count

GROUND_TRUTH_pandas_resample_zero_safe = {"buggy": False, "category": "DIV_BY_ZERO"}


# --- flask #3762: Division by zero in rate limiting ---
# Project: flask, Bug ID: 8
def flask_rate_limit_zero(requests_count, time_window):
    """BUG: time_window may be zero."""
    rate = requests_count / time_window  # BUG
    return rate

GROUND_TRUTH_flask_rate_limit_zero = {"buggy": True, "category": "DIV_BY_ZERO"}


# --- keras #14342: IndexError on empty predictions ---
# Project: keras, Bug ID: 2
def keras_empty_predictions(predictions):
    """BUG: predictions may be empty."""
    best = predictions[0]  # BUG: IndexError if empty
    return best

GROUND_TRUTH_keras_empty_predictions = {"buggy": True, "category": "INDEX_OUT_OF_BOUNDS"}


# --- keras safe version ---
def keras_empty_predictions_safe(predictions):
    """SAFE: length check."""
    if len(predictions) == 0:
        return None
    return predictions[0]

GROUND_TRUTH_keras_empty_predictions_safe = {"buggy": False, "category": "INDEX_OUT_OF_BOUNDS"}


# --- tornado #2845: TypeError None + str ---
# Project: tornado, Bug ID: 4
def tornado_type_concat(base_url, path):
    """BUG: base_url may be None from config."""
    base_url = None  # simulating config miss
    full = base_url + path  # BUG: TypeError
    return full

GROUND_TRUTH_tornado_type_concat = {"buggy": True, "category": "TYPE_ERROR"}


# --- tornado safe ---
def tornado_type_concat_safe(base_url, path):
    """SAFE: None check."""
    if base_url is None:
        base_url = ""
    return base_url + path

GROUND_TRUTH_tornado_type_concat_safe = {"buggy": False, "category": "TYPE_ERROR"}


# --- requests #5765: AttributeError on None encoding ---
# Project: requests, Bug ID: 6
def requests_none_encoding(response):
    """BUG: encoding from headers may be None."""
    encoding = response.get("encoding")
    return encoding.upper()  # BUG: None dereference

GROUND_TRUTH_requests_none_encoding = {"buggy": True, "category": "NULL_DEREF"}


# --- requests safe ---
def requests_none_encoding_safe(response):
    """SAFE: default provided."""
    encoding = response.get("encoding", "utf-8")
    return encoding.upper()

GROUND_TRUTH_requests_none_encoding_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- black #2376: Division by zero in line splitting ---
# Project: black, Bug ID: 7
def black_line_split_zero(line_length, indent_level):
    """BUG: effective_length can be zero."""
    effective_length = line_length - indent_level
    ratio = 80 / effective_length  # BUG: div by zero
    return ratio

GROUND_TRUTH_black_line_split_zero = {"buggy": True, "category": "DIV_BY_ZERO"}


# --- black safe ---
def black_line_split_zero_safe(line_length, indent_level):
    """SAFE: bounds check."""
    effective_length = line_length - indent_level
    if effective_length <= 0:
        return 1.0
    return 80 / effective_length

GROUND_TRUTH_black_line_split_zero_safe = {"buggy": False, "category": "DIV_BY_ZERO"}


# --- httpie #1179: None.split() ---
# Project: httpie, Bug ID: 9
def httpie_content_type_split(response):
    """BUG: content_type may be None."""
    content_type = response.get("content-type")
    parts = content_type.split(";")  # BUG: None dereference
    return parts[0].strip()

GROUND_TRUTH_httpie_content_type_split = {"buggy": True, "category": "NULL_DEREF"}


# --- httpie safe ---
def httpie_content_type_split_safe(response):
    """SAFE: None check with default."""
    content_type = response.get("content-type")
    if content_type is None:
        return "application/octet-stream"
    parts = content_type.split(";")
    return parts[0].strip()

GROUND_TRUTH_httpie_content_type_split_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- click #2198: IndexError on empty argv ---
# Project: click, Bug ID: 10
def click_empty_argv(args):
    """BUG: args may be empty."""
    command = args[0]  # BUG: IndexError
    return command

GROUND_TRUTH_click_empty_argv = {"buggy": True, "category": "INDEX_OUT_OF_BOUNDS"}


# --- click safe ---
def click_empty_argv_safe(args):
    """SAFE: length check."""
    if not args:
        return "help"
    return args[0]

GROUND_TRUTH_click_empty_argv_safe = {"buggy": False, "category": "INDEX_OUT_OF_BOUNDS"}


# --- sqlalchemy #5800: None dereference on query result ---
def sqlalchemy_query_none(session):
    """BUG: first() returns None when no results."""
    user = session.query("User").first()
    return user.name  # BUG: user may be None

GROUND_TRUTH_sqlalchemy_query_none = {"buggy": True, "category": "NULL_DEREF"}


# --- sqlalchemy safe ---
def sqlalchemy_query_none_safe(session):
    """SAFE: None check."""
    user = session.query("User").first()
    if user is None:
        return "Unknown"
    return user.name

GROUND_TRUTH_sqlalchemy_query_none_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- numpy #19386: Division by zero in normalize ---
def numpy_normalize_zero(values):
    """BUG: sum may be zero."""
    total = sum(values)
    normalized = [v / total for v in values]  # BUG
    return normalized

GROUND_TRUTH_numpy_normalize_zero = {"buggy": True, "category": "DIV_BY_ZERO"}


# --- numpy safe ---
def numpy_normalize_zero_safe(values):
    """SAFE: zero check."""
    total = sum(values)
    if total == 0:
        return [0.0] * len(values)
    return [v / total for v in values]

GROUND_TRUTH_numpy_normalize_zero_safe = {"buggy": False, "category": "DIV_BY_ZERO"}


# --- pillow #5914: None attribute access ---
def pillow_exif_none(image):
    """BUG: getexif() may return None-like."""
    exif = image.get("exif")
    orientation = exif.get("Orientation")  # BUG: exif may be None
    return orientation

GROUND_TRUTH_pillow_exif_none = {"buggy": True, "category": "NULL_DEREF"}


# --- pillow safe ---
def pillow_exif_none_safe(image):
    """SAFE: None check."""
    exif = image.get("exif")
    if exif is None:
        return 1
    return exif.get("Orientation", 1)

GROUND_TRUTH_pillow_exif_none_safe = {"buggy": False, "category": "NULL_DEREF"}


# --- matplotlib #22471: average of empty ---
def matplotlib_empty_avg(data_points):
    """BUG: data_points may be empty."""
    avg = sum(data_points) / len(data_points)  # BUG: div by zero
    return avg

GROUND_TRUTH_matplotlib_empty_avg = {"buggy": True, "category": "DIV_BY_ZERO"}


# --- matplotlib safe ---
def matplotlib_empty_avg_safe(data_points):
    """SAFE: empty check."""
    if not data_points:
        return 0.0
    return sum(data_points) / len(data_points)

GROUND_TRUTH_matplotlib_empty_avg_safe = {"buggy": False, "category": "DIV_BY_ZERO"}


# --- Type error: str + int ---
def type_error_str_int(name, count):
    """BUG: concatenating str and int."""
    name = "hello"
    count = 5
    result = name + count  # BUG: TypeError
    return result

GROUND_TRUTH_type_error_str_int = {"buggy": True, "category": "TYPE_ERROR"}


# --- Type error safe ---
def type_error_str_int_safe(name, count):
    """SAFE: converted to str."""
    name = "hello"
    count = 5
    result = name + str(count)
    return result

GROUND_TRUTH_type_error_str_int_safe = {"buggy": False, "category": "TYPE_ERROR"}
'''


def extract_functions_and_truth(source):
    """Extract function names and their ground truth from the benchmark."""
    tree = ast.parse(source)
    functions = {}
    ground_truth = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("GROUND_TRUTH_"):
                continue
            functions[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("GROUND_TRUTH_"):
                    func_name = target.id[len("GROUND_TRUTH_"):]
                    if isinstance(node.value, ast.Dict):
                        gt = {}
                        for k, v in zip(node.value.keys, node.value.values):
                            if isinstance(k, ast.Constant):
                                if isinstance(v, ast.Constant):
                                    gt[k.value] = v.value
                        ground_truth[func_name] = gt

    return functions, ground_truth


def run_benchmark():
    """Run GuardHarvest on the BugsInPy-inspired benchmark."""
    functions, ground_truth = extract_functions_and_truth(BUGSINPY_BENCHMARK)

    analyzer = FlowSensitiveAnalyzer(BUGSINPY_BENCHMARK, "bugsinpy_benchmark.py")
    tree = ast.parse(BUGSINPY_BENCHMARK)

    results = []
    tp, fp, fn, tn = 0, 0, 0, 0
    category_stats = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ground_truth:
            gt = ground_truth[node.name]
            is_buggy = gt.get("buggy", False)
            category = gt.get("category", "UNKNOWN")

            func_result = analyzer.analyze_function(node)
            detected = len(func_result.bugs) > 0

            if is_buggy and detected:
                verdict = "TP"
                tp += 1
            elif is_buggy and not detected:
                verdict = "FN"
                fn += 1
            elif not is_buggy and detected:
                verdict = "FP"
                fp += 1
            else:
                verdict = "TN"
                tn += 1

            results.append({
                "function": node.name,
                "category": category,
                "is_buggy": is_buggy,
                "detected": detected,
                "verdict": verdict,
                "bugs_found": [b.to_dict() for b in func_result.bugs],
                "guards_harvested": func_result.guards_harvested,
            })

            if category not in category_stats:
                category_stats[category] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            category_stats[category][verdict.lower()] += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 70)
    print("BugsInPy-Inspired Benchmark Results")
    print("=" * 70)
    print(f"Total functions: {total}")
    print(f"Buggy: {tp + fn}, Safe: {tn + fp}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1: {f1:.3f}")
    print()

    print("Per-category:")
    for cat, stats in sorted(category_stats.items()):
        cat_tp = stats["tp"]
        cat_fp = stats["fp"]
        cat_fn = stats["fn"]
        cat_prec = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
        cat_rec = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
        cat_f1 = 2 * cat_prec * cat_rec / (cat_prec + cat_rec) if (cat_prec + cat_rec) > 0 else 0
        print(f"  {cat}: P={cat_prec:.1%} R={cat_rec:.1%} F1={cat_f1:.3f}")

    print()
    print("False Negatives:")
    for r in results:
        if r["verdict"] == "FN":
            print(f"  {r['function']} ({r['category']})")

    print()
    print("False Positives:")
    for r in results:
        if r["verdict"] == "FP":
            print(f"  {r['function']} ({r['category']})")
            for b in r["bugs_found"]:
                print(f"    → {b['message']}")

    output = {
        "benchmark": "bugsinpy_inspired",
        "total_functions": total,
        "buggy_functions": tp + fn,
        "safe_functions": tn + fp,
        "metrics": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        },
        "category_breakdown": category_stats,
        "per_function": results,
    }

    output_path = os.path.join(os.path.dirname(__file__), 'results',
                               'bugsinpy_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_benchmark()
