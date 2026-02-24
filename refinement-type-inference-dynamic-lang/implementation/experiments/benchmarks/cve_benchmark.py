"""
CVE and Real-World Bug Benchmark Suite.

150 functions capturing real bug patterns from:
- CVE reports affecting Python packages (requests, Django, Flask, Pillow, etc.)
- BugsInPy dataset bug descriptions
- Top StackOverflow Python bug questions
- GitHub issue trackers of major projects

Each function is a MINIMAL reproduction of the bug *pattern* in fresh code.
No verbatim code is copied. Ground truth is assigned by the bug pattern.

Categories:
  NULL_DEREF    - None dereference / attribute on None
  DIV_BY_ZERO   - Division by zero
  INDEX_OOB     - Index out of bounds
  TYPE_ERROR    - Type mismatch operations
  TAINT_FLOW    - Tainted input reaching sensitive sink
  UNGUARDED_OPT - Optional value used without check
"""

# ==============================================================
# Ground truth: maps function name -> {has_bug, category, source}
# ==============================================================
GROUND_TRUTH = {
    # --- NULL_DEREF patterns (30 buggy + 15 safe = 45) ---
    "cve_2023_requests_redirect": {"has_bug": True, "category": "NULL_DEREF",
        "source": "requests redirect history pattern"},
    "cve_2022_flask_config": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Flask config.get() pattern"},
    "cve_django_queryset": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Django queryset.first() pattern"},
    "cve_pillow_getattr": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Pillow image attribute access pattern"},
    "bugsinpy_pandas_iloc": {"has_bug": True, "category": "NULL_DEREF",
        "source": "pandas iloc on empty DataFrame pattern"},
    "bugsinpy_tornado_header": {"has_bug": True, "category": "NULL_DEREF",
        "source": "tornado request header pattern"},
    "so_json_loads_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "json.loads returning None on malformed input"},
    "so_dict_get_chain": {"has_bug": True, "category": "NULL_DEREF",
        "source": "chained dict.get() returning None"},
    "so_list_find_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "str.find() returning -1 used as index"},
    "so_re_match_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "re.match() returning None"},
    "github_click_param": {"has_bug": True, "category": "NULL_DEREF",
        "source": "click parameter default None pattern"},
    "github_sqlalchemy_first": {"has_bug": True, "category": "NULL_DEREF",
        "source": "SQLAlchemy query.first() returns None"},
    "github_celery_result": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Celery AsyncResult.result can be None"},
    "github_boto3_response": {"has_bug": True, "category": "NULL_DEREF",
        "source": "boto3 response can lack expected keys"},
    "github_yaml_load": {"has_bug": True, "category": "NULL_DEREF",
        "source": "yaml.safe_load returns None on empty doc"},
    "pathlib_parent_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "pathlib resolve can return None in edge cases"},
    "configparser_default": {"has_bug": True, "category": "NULL_DEREF",
        "source": "configparser fallback-less get returns None"},
    "environ_get_split": {"has_bug": True, "category": "NULL_DEREF",
        "source": "os.environ.get().split() on None"},
    "socket_recv_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "socket recv returning empty on closed connection"},
    "xml_find_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "xml.etree find() returns None"},
    "csv_next_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "next() with default None"},
    "weakref_deref_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "weakref callback on dead object"},
    "logging_handler_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "logging handler stream can be None"},
    "multiprocessing_result": {"has_bug": True, "category": "NULL_DEREF",
        "source": "multiprocessing Pool result can be None"},
    "argparse_parse_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "argparse attribute not always set"},
    "http_header_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "HTTP header lookup returning None"},
    "subprocess_stdout": {"has_bug": True, "category": "NULL_DEREF",
        "source": "subprocess.run stdout is None without capture"},
    "class_attr_init_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "class attribute initialized as None in __init__"},
    "optional_callback_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "optional callback function not checked"},
    "nested_dict_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "nested dict access with intermediate None"},

    # Safe None-handling patterns (15)
    "safe_dict_get_default": {"has_bug": False, "category": "NULL_DEREF",
        "source": "dict.get with non-None default"},
    "safe_isinstance_guard": {"has_bug": False, "category": "NULL_DEREF",
        "source": "isinstance check before method call"},
    "safe_none_check_attr": {"has_bug": False, "category": "NULL_DEREF",
        "source": "explicit None check before attribute access"},
    "safe_or_default": {"has_bug": False, "category": "NULL_DEREF",
        "source": "x or default_value pattern"},
    "safe_ternary_none": {"has_bug": False, "category": "NULL_DEREF",
        "source": "ternary with None check"},
    "safe_walrus_guard": {"has_bug": False, "category": "NULL_DEREF",
        "source": "walrus operator with truthiness check"},
    "safe_try_except_attr": {"has_bug": False, "category": "NULL_DEREF",
        "source": "try/except around potentially None access"},
    "safe_assert_not_none": {"has_bug": False, "category": "NULL_DEREF",
        "source": "assert x is not None guard"},
    "safe_early_return": {"has_bug": False, "category": "NULL_DEREF",
        "source": "early return on None check"},
    "safe_truthiness_check": {"has_bug": False, "category": "NULL_DEREF",
        "source": "if x: pattern for None check"},
    "safe_hasattr_guard": {"has_bug": False, "category": "NULL_DEREF",
        "source": "hasattr check before getattr"},
    "safe_conditional_assign": {"has_bug": False, "category": "NULL_DEREF",
        "source": "conditional assignment before use"},
    "safe_getattr_default": {"has_bug": False, "category": "NULL_DEREF",
        "source": "getattr with default value"},
    "safe_chain_short_circuit": {"has_bug": False, "category": "NULL_DEREF",
        "source": "x and x.attr short-circuit pattern"},
    "safe_multi_guard": {"has_bug": False, "category": "NULL_DEREF",
        "source": "multiple guards combined with and"},

    # --- DIV_BY_ZERO patterns (20 buggy + 10 safe = 30) ---
    "cve_stats_mean": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "statistics.mean on empty list"},
    "cve_normalize_vector": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "vector normalization with zero magnitude"},
    "bugsinpy_avg_grade": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "average computation without empty check"},
    "bugsinpy_pct_change": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "percentage change from zero baseline"},
    "so_weighted_avg": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "weighted average with zero total weight"},
    "so_rate_computation": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "rate = count/time with time=0"},
    "github_batch_size": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "num_batches = total // batch_size with batch_size=0"},
    "github_resize_ratio": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "image resize ratio with zero dimension"},
    "github_tax_rate": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "tax computation with zero income"},
    "github_learning_rate": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "learning rate decay with zero step"},
    "modulo_zero": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "modulo by zero"},
    "floor_div_zero": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "floor division by zero"},
    "div_by_subtraction": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "division by result of subtraction that can be zero"},
    "div_by_len_empty": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "division by len(list) where list can be empty"},
    "div_by_count": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "division by count that can be zero"},
    "harmonic_mean_zero": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "harmonic mean with zero element"},
    "variance_one_elem": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "sample variance with n=1"},
    "ratio_update": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "ratio update with denominator from subtraction"},
    "probability_normalize": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "probability normalization with zero sum"},
    "fps_counter": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "FPS = frames / elapsed_time with elapsed=0"},

    # Safe div-by-zero patterns (10)
    "safe_div_check": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "explicit zero check before division"},
    "safe_div_max": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "max(denominator, 1) pattern"},
    "safe_div_or_default": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "denominator or default_value pattern"},
    "safe_div_epsilon": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "division by x + epsilon pattern"},
    "safe_div_ternary": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "result = a/b if b != 0 else 0 pattern"},
    "safe_div_guard_len": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "if len(items) > 0: avg = sum/len"},
    "safe_div_try_except": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "try/except ZeroDivisionError"},
    "safe_div_abs_positive": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "division by abs(x) + 1"},
    "safe_div_constant": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "division by known non-zero constant"},
    "safe_div_squared": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "division by x*x + 1"},

    # --- INDEX_OOB patterns (15 buggy + 8 safe = 23) ---
    "bugsinpy_list_pop_empty": {"has_bug": True, "category": "INDEX_OOB",
        "source": "pop on potentially empty list"},
    "bugsinpy_argv_access": {"has_bug": True, "category": "INDEX_OOB",
        "source": "sys.argv[1] without length check"},
    "so_split_fixed_index": {"has_bug": True, "category": "INDEX_OOB",
        "source": "str.split()[n] assuming n parts"},
    "so_csv_column_access": {"has_bug": True, "category": "INDEX_OOB",
        "source": "CSV row[col] assuming column exists"},
    "github_first_element": {"has_bug": True, "category": "INDEX_OOB",
        "source": "list[0] without empty check"},
    "github_last_element": {"has_bug": True, "category": "INDEX_OOB",
        "source": "list[-1] without empty check"},
    "github_zip_shortest": {"has_bug": True, "category": "INDEX_OOB",
        "source": "accessing index beyond shorter list in zip"},
    "tuple_unpack_mismatch": {"has_bug": True, "category": "INDEX_OOB",
        "source": "tuple unpacking with wrong number of elements"},
    "deque_pop_empty": {"has_bug": True, "category": "INDEX_OOB",
        "source": "deque popleft on empty deque"},
    "matrix_access_oob": {"has_bug": True, "category": "INDEX_OOB",
        "source": "matrix[i][j] without bounds check"},
    "circular_buffer_oob": {"has_bug": True, "category": "INDEX_OOB",
        "source": "circular buffer index without modulo"},
    "string_char_access": {"has_bug": True, "category": "INDEX_OOB",
        "source": "string[i] without length check"},
    "queue_get_empty": {"has_bug": True, "category": "INDEX_OOB",
        "source": "queue get from empty queue"},
    "stack_peek_empty": {"has_bug": True, "category": "INDEX_OOB",
        "source": "stack[-1] peek on empty stack"},
    "sliding_window_oob": {"has_bug": True, "category": "INDEX_OOB",
        "source": "sliding window access beyond end"},

    # Safe index patterns (8)
    "safe_index_len_check": {"has_bug": False, "category": "INDEX_OOB",
        "source": "len check before index access"},
    "safe_index_try_except": {"has_bug": False, "category": "INDEX_OOB",
        "source": "try/except IndexError"},
    "safe_index_enumerate": {"has_bug": False, "category": "INDEX_OOB",
        "source": "enumerate-based access"},
    "safe_index_slice": {"has_bug": False, "category": "INDEX_OOB",
        "source": "slice that never raises"},
    "safe_index_truthiness": {"has_bug": False, "category": "INDEX_OOB",
        "source": "if items: items[0]"},
    "safe_index_modulo": {"has_bug": False, "category": "INDEX_OOB",
        "source": "index % len pattern"},
    "safe_index_min_len": {"has_bug": False, "category": "INDEX_OOB",
        "source": "min(i, len-1) pattern"},
    "safe_index_bounded_loop": {"has_bug": False, "category": "INDEX_OOB",
        "source": "for i in range(len(items)): items[i]"},

    # --- TYPE_ERROR patterns (15 buggy + 7 safe = 22) ---
    "type_str_plus_int": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "string + integer concatenation"},
    "type_none_plus_int": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "None + integer arithmetic"},
    "type_list_plus_int": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "list + integer"},
    "type_str_subtract": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "string subtraction"},
    "type_dict_multiply": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "dict * integer"},
    "type_bool_concat_str": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "bool + str concatenation"},
    "type_none_subscript": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "None[index] subscript"},
    "type_int_subscript": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "integer[index] subscript"},
    "type_none_iteration": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "for x in None iteration"},
    "type_int_len": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "len(integer) call"},
    "type_float_index": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "list[float] index"},
    "type_none_comparison_op": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "None > integer comparison"},
    "type_str_modulo_list": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "string % list formatting error"},
    "type_set_subscript": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "set[index] subscript"},
    "type_none_call": {"has_bug": True, "category": "TYPE_ERROR",
        "source": "None() call"},

    # Safe type patterns (7)
    "safe_type_str_cast": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "str(x) + string pattern"},
    "safe_type_isinstance_guard": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "isinstance check before operation"},
    "safe_type_numeric_tower": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "int + float = float (valid)"},
    "safe_type_str_multiply": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "string * int (valid repeat)"},
    "safe_type_list_extend": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "list + list (valid concat)"},
    "safe_type_bool_arithmetic": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "bool + int (valid, bool subclass of int)"},
    "safe_type_dict_update": {"has_bug": False, "category": "TYPE_ERROR",
        "source": "dict | dict (valid merge)"},

    # --- TAINT_FLOW patterns (10 buggy + 5 safe = 15) ---
    "taint_sql_format": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "SQL injection via f-string"},
    "taint_sql_percent": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "SQL injection via % formatting"},
    "taint_os_system": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "command injection via os.system"},
    "taint_subprocess_shell": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "command injection via subprocess shell=True"},
    "taint_eval_input": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "code injection via eval(input())"},
    "taint_open_path": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "path traversal via open(user_path)"},
    "taint_yaml_load": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "unsafe YAML deserialization"},
    "taint_pickle_loads": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "unsafe pickle deserialization"},
    "taint_exec_string": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "code injection via exec()"},
    "taint_template_render": {"has_bug": True, "category": "TAINT_FLOW",
        "source": "XSS via unescaped template rendering"},

    # Safe taint patterns (5)
    "safe_taint_parameterized": {"has_bug": False, "category": "TAINT_FLOW",
        "source": "parameterized SQL query"},
    "safe_taint_sanitize_int": {"has_bug": False, "category": "TAINT_FLOW",
        "source": "int() sanitization"},
    "safe_taint_allowlist": {"has_bug": False, "category": "TAINT_FLOW",
        "source": "allowlist validation before use"},
    "safe_taint_escape": {"has_bug": False, "category": "TAINT_FLOW",
        "source": "html.escape before rendering"},
    "safe_taint_subprocess_list": {"has_bug": False, "category": "TAINT_FLOW",
        "source": "subprocess with list args (no shell)"},
}


# ==============================================================
# Function implementations
# ==============================================================

# --- NULL_DEREF buggy functions ---

def cve_2023_requests_redirect(session, url):
    """Pattern from requests: redirect without checking response."""
    resp = session.get(url)
    location = resp.headers.get("Location")
    return location.strip()

def cve_2022_flask_config(app, key):
    """Pattern from Flask: config access without None check."""
    value = app.config.get(key)
    return value.lower()

def cve_django_queryset(items, pk):
    """Pattern from Django: queryset.first() returns None."""
    filtered = [x for x in items if x.get("id") == pk]
    obj = filtered[0] if filtered else None
    return obj["name"]

def cve_pillow_getattr(image, attr_name):
    """Pattern from Pillow: image attribute access on closed image."""
    info = getattr(image, "info", None)
    return info["dpi"]

def bugsinpy_pandas_iloc(data, idx):
    """Pattern from pandas: iloc on potentially empty result."""
    filtered = [x for x in data if x > 0]
    result = filtered[0] if filtered else None
    return result * 2

def bugsinpy_tornado_header(headers, name):
    """Pattern from tornado: header lookup returning None."""
    value = headers.get(name)
    return value.decode("utf-8")

def so_json_loads_none(raw_text):
    """Pattern: json.loads on None input."""
    import json
    data = None
    if raw_text:
        try:
            data = json.loads(raw_text)
        except Exception:
            pass
    return data["key"]

def so_dict_get_chain(config, section, key):
    """Pattern: chained dict.get() returning None."""
    sec = config.get(section)
    return sec.get(key)

def so_list_find_none(text, target):
    """Pattern: str.find() returning -1 treated as valid index."""
    idx = text.find(target)
    return text[idx:idx+len(target)]

def so_re_match_none(pattern, text):
    """Pattern: re.match returning None on no match."""
    import re
    m = re.match(pattern, text)
    return m.group(0)

def github_click_param(ctx, param_name):
    """Pattern from click: parameter default None."""
    value = ctx.params.get(param_name)
    return value.upper()

def github_sqlalchemy_first(query_result):
    """Pattern from SQLAlchemy: query.first() returns None."""
    items = query_result if query_result else []
    obj = items[0] if items else None
    return obj.id

def github_celery_result(task_id, results):
    """Pattern from Celery: AsyncResult.result can be None."""
    result = results.get(task_id)
    return result.status

def github_boto3_response(response):
    """Pattern from boto3: response missing expected keys."""
    body = response.get("Body")
    return body.read()

def github_yaml_load(yaml_text):
    """Pattern: yaml.safe_load returns None on empty document."""
    import yaml
    data = None
    if yaml_text:
        data = yaml.safe_load(yaml_text) if hasattr(yaml, 'safe_load') else None
    return data["config"]

def pathlib_parent_none(file_path):
    """Pattern: pathlib operations returning None in edge cases."""
    from pathlib import Path
    p = Path(file_path)
    parent = p.parent if str(p) != "." else None
    return parent.name

def configparser_default(config, section, key):
    """Pattern: configparser get without fallback."""
    value = config.get(key)
    return int(value)

def environ_get_split(var_name):
    """Pattern: os.environ.get().split() on None."""
    import os
    path = os.environ.get(var_name)
    return path.split(":")

def socket_recv_none(data_buffer):
    """Pattern: recv returning None/empty on closed connection."""
    data = data_buffer.get("payload")
    return data.decode("utf-8")

def xml_find_none(tree, tag):
    """Pattern: xml.etree find() returns None."""
    element = tree.find(tag) if hasattr(tree, 'find') else None
    if not element:
        element = None
    return element.text

def csv_next_none(rows, default=None):
    """Pattern: next() with default None used unsafely."""
    row = next(iter(rows), None)
    return row[0]

def weakref_deref_none(ref_dict, key):
    """Pattern: weakref to dead object."""
    obj = ref_dict.get(key)
    return obj.value

def logging_handler_none(handler_config):
    """Pattern: logging handler stream can be None."""
    stream = handler_config.get("stream")
    stream.write("log message\n")

def multiprocessing_result(pool_results, idx):
    """Pattern: multiprocessing Pool result can be None."""
    result = pool_results.get(idx)
    return result.get()

def argparse_parse_none(args, attr):
    """Pattern: argparse attribute not always set."""
    value = getattr(args, attr, None)
    return value.strip()

def http_header_none(headers, name):
    """Pattern: HTTP header lookup returning None."""
    value = headers.get(name)
    parts = value.split(",")
    return parts

def subprocess_stdout(result):
    """Pattern: subprocess.run stdout is None without capture."""
    stdout = result.get("stdout")
    return stdout.decode("utf-8")

def class_attr_init_none(obj):
    """Pattern: class attribute initialized as None in __init__."""
    connection = getattr(obj, "connection", None)
    return connection.execute("SELECT 1")

def optional_callback_none(callbacks, event):
    """Pattern: optional callback function not checked."""
    handler = callbacks.get(event)
    return handler("event data")

def nested_dict_none(data, key1, key2):
    """Pattern: nested dict access with intermediate None."""
    inner = data.get(key1)
    return inner[key2]


# --- NULL_DEREF safe functions ---

def safe_dict_get_default(d, key):
    value = d.get(key, "default")
    return value.upper()

def safe_isinstance_guard(obj):
    if isinstance(obj, str):
        return obj.upper()
    return ""

def safe_none_check_attr(config, key):
    value = config.get(key)
    if value is not None:
        return value.strip()
    return ""

def safe_or_default(config, key):
    value = config.get(key) or "default"
    return value.upper()

def safe_ternary_none(config, key):
    value = config.get(key)
    return value.strip() if value is not None else ""

def safe_walrus_guard(items):
    if (first := next(iter(items), None)) is not None:
        return first.upper()
    return ""

def safe_try_except_attr(obj, attr):
    try:
        return getattr(obj, attr).upper()
    except (AttributeError, TypeError):
        return ""

def safe_assert_not_none(config, key):
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required key: {key}")
    return value.strip()

def safe_early_return(config, key):
    value = config.get(key)
    if value is None:
        return ""
    return value.strip()

def safe_truthiness_check(config, key):
    value = config.get(key)
    if value:
        return value.strip()
    return ""

def safe_hasattr_guard(obj, attr):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return None

def safe_conditional_assign(items, idx):
    value = items[idx] if idx < len(items) else "default"
    return value.upper()

def safe_getattr_default(obj, attr):
    value = getattr(obj, attr, "default")
    return value.upper()

def safe_chain_short_circuit(obj):
    result = obj and obj.get("key")
    if result:
        return result.upper()
    return ""

def safe_multi_guard(config, key):
    value = config.get(key)
    if value is not None and isinstance(value, str):
        return value.strip()
    return ""


# --- DIV_BY_ZERO buggy functions ---

def cve_stats_mean(values):
    """Division by len of potentially empty list."""
    total = sum(values)
    return total / len(values)

def cve_normalize_vector(vec):
    """Vector normalization with zero magnitude."""
    magnitude = sum(x*x for x in vec) ** 0.5
    return [x / magnitude for x in vec]

def bugsinpy_avg_grade(grades):
    """Average grade without empty check."""
    return sum(grades) / len(grades)

def bugsinpy_pct_change(old_value, new_value):
    """Percentage change from zero baseline."""
    return (new_value - old_value) / old_value * 100

def so_weighted_avg(values, weights):
    """Weighted average with zero total weight."""
    total_weight = sum(weights)
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def so_rate_computation(count, elapsed_seconds):
    """Rate computation with potentially zero time."""
    return count / elapsed_seconds

def github_batch_size(total_items, batch_size):
    """Batch computation with zero batch size."""
    num_batches = total_items // batch_size
    return num_batches

def github_resize_ratio(width, height, target_width):
    """Image resize ratio with zero dimension."""
    ratio = target_width / width
    new_height = int(height * ratio)
    return new_height

def github_tax_rate(income, tax_amount):
    """Tax computation with zero income."""
    effective_rate = tax_amount / income
    return effective_rate * 100

def github_learning_rate(initial_lr, step, decay_steps):
    """Learning rate decay with zero step."""
    return initial_lr * (0.96 ** (step / decay_steps))

def modulo_zero(value, divisor):
    """Modulo by zero."""
    return value % divisor

def floor_div_zero(total, groups):
    """Floor division by zero."""
    per_group = total // groups
    return per_group

def div_by_subtraction(a, b):
    """Division by subtraction result that can be zero."""
    diff = a - b
    return 100 / diff

def div_by_len_empty(items):
    """Division by len of potentially empty list."""
    return sum(items) / len(items)

def div_by_count(data, predicate_count):
    """Division by count that can be zero."""
    return sum(data) / predicate_count

def harmonic_mean_zero(values):
    """Harmonic mean with zero element."""
    reciprocal_sum = sum(1/v for v in values)
    return len(values) / reciprocal_sum

def variance_one_elem(values):
    """Sample variance with n=1 causes division by zero."""
    mean = sum(values) / len(values)
    sq_diff = sum((x - mean) ** 2 for x in values)
    return sq_diff / (len(values) - 1)

def ratio_update(current, previous, smoothing):
    """Ratio update with denominator from subtraction."""
    delta = current - previous
    return smoothing / delta

def probability_normalize(probs):
    """Probability normalization with zero sum."""
    total = sum(probs)
    return [p / total for p in probs]

def fps_counter(frame_count, start_time, current_time):
    """FPS computation with zero elapsed time."""
    elapsed = current_time - start_time
    return frame_count / elapsed


# --- DIV_BY_ZERO safe functions ---

def safe_div_check(a, b):
    if b != 0:
        return a / b
    return 0

def safe_div_max(total, count):
    return total / max(count, 1)

def safe_div_or_default(total, count):
    divisor = count or 1
    return total / divisor

def safe_div_epsilon(a, b):
    return a / (b + 1e-10)

def safe_div_ternary(a, b):
    return a / b if b != 0 else 0.0

def safe_div_guard_len(items):
    if len(items) > 0:
        return sum(items) / len(items)
    return 0.0

def safe_div_try_except(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0

def safe_div_abs_positive(a, b):
    return a / (abs(b) + 1)

def safe_div_constant(total):
    return total / 100

def safe_div_squared(a, b):
    return a / (b * b + 1)


# --- INDEX_OOB buggy functions ---

def bugsinpy_list_pop_empty(items):
    """Pop on potentially empty list."""
    return items.pop()

def bugsinpy_argv_access(argv):
    """sys.argv[1] without length check."""
    return argv[1]

def so_split_fixed_index(text, delimiter, idx):
    """str.split()[n] assuming n parts exist."""
    parts = text.split(delimiter)
    return parts[idx]

def so_csv_column_access(row, col_idx):
    """CSV row[col] assuming column exists."""
    return row[col_idx]

def github_first_element(items):
    """list[0] without empty check."""
    return items[0]

def github_last_element(items):
    """list[-1] without empty check."""
    return items[-1]

def github_zip_shortest(list_a, list_b, idx):
    """Accessing index beyond shorter list."""
    return list_a[idx] + list_b[idx]

def tuple_unpack_mismatch(items):
    """Tuple unpacking with wrong count."""
    a, b, c = items
    return a + b + c

def deque_pop_empty(dq):
    """Deque popleft on empty deque."""
    return dq.popleft()

def matrix_access_oob(matrix, i, j):
    """Matrix access without bounds check."""
    return matrix[i][j]

def circular_buffer_oob(buffer, idx):
    """Circular buffer without modulo."""
    return buffer[idx]

def string_char_access(text, pos):
    """String character access without length check."""
    return text[pos]

def queue_get_empty(queue):
    """Queue get from empty queue."""
    return queue.pop(0)

def stack_peek_empty(stack):
    """Stack peek on empty stack."""
    return stack[-1]

def sliding_window_oob(data, start, window_size):
    """Sliding window access beyond end."""
    return [data[start + i] for i in range(window_size)]


# --- INDEX_OOB safe functions ---

def safe_index_len_check(items, idx):
    if idx < len(items):
        return items[idx]
    return None

def safe_index_try_except(items, idx):
    try:
        return items[idx]
    except IndexError:
        return None

def safe_index_enumerate(items):
    for i, item in enumerate(items):
        return item
    return None

def safe_index_slice(items, start, end):
    return items[start:end]

def safe_index_truthiness(items):
    if items:
        return items[0]
    return None

def safe_index_modulo(items, idx):
    if items:
        return items[idx % len(items)]
    return None

def safe_index_min_len(items, idx):
    safe_idx = min(idx, len(items) - 1) if items else 0
    return items[safe_idx] if items else None

def safe_index_bounded_loop(items):
    result = []
    for i in range(len(items)):
        result.append(items[i])
    return result


# --- TYPE_ERROR buggy functions ---

def type_str_plus_int(name, count):
    """String + integer concatenation."""
    name = "hello"
    count = 42
    return name + count

def type_none_plus_int(value):
    """None + integer arithmetic."""
    x = None
    return x + 1

def type_list_plus_int(items):
    """List + integer."""
    items = [1, 2, 3]
    return items + 1

def type_str_subtract(a, b):
    """String subtraction."""
    a = "hello"
    b = "world"
    return a - b

def type_dict_multiply(d, n):
    """Dict * integer."""
    d = {"a": 1}
    return d * 3

def type_bool_concat_str(flag, text):
    """Bool + str concatenation."""
    flag = True
    text = " is active"
    return flag + text

def type_none_subscript(x):
    """None[index] subscript."""
    x = None
    return x[0]

def type_int_subscript(x):
    """Integer subscript."""
    x = 42
    return x[0]

def type_none_iteration(items):
    """Iteration over None."""
    items = None
    for x in items:
        pass
    return x

def type_int_len(x):
    """len(integer) call."""
    x = 42
    return len(x)

def type_float_index(items, idx):
    """list[float] index."""
    items = [1, 2, 3]
    idx = 1.5
    return items[idx]

def type_none_comparison_op(x, y):
    """None > integer comparison."""
    x = None
    y = 5
    return x > y

def type_str_modulo_list(fmt, args):
    """String % list formatting error."""
    fmt = "%s %s"
    args = [1, 2, 3]
    return fmt % args

def type_set_subscript(s, idx):
    """Set subscript."""
    s = {1, 2, 3}
    return s[0]

def type_none_call(func):
    """None() call."""
    func = None
    return func()


# --- TYPE_ERROR safe functions ---

def safe_type_str_cast(name, count):
    return str(name) + str(count)

def safe_type_isinstance_guard(value):
    if isinstance(value, int):
        return value + 1
    return 0

def safe_type_numeric_tower(a, b):
    a = 1
    b = 2.5
    return a + b

def safe_type_str_multiply(text, n):
    text = "ha"
    n = 3
    return text * n

def safe_type_list_extend(a, b):
    a = [1, 2]
    b = [3, 4]
    return a + b

def safe_type_bool_arithmetic(flag, value):
    flag = True
    value = 5
    return flag + value

def safe_type_dict_update(a, b):
    a = {"x": 1}
    b = {"y": 2}
    a.update(b)
    return a


# --- TAINT_FLOW buggy functions ---

def taint_sql_format(user_input):
    """SQL injection via f-string."""
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query

def taint_sql_percent(user_input):
    """SQL injection via % formatting."""
    query = "SELECT * FROM users WHERE name = '%s'" % user_input
    return query

def taint_os_system(user_input):
    """Command injection via os.system."""
    import os
    cmd = "ls " + user_input
    os.system(cmd)

def taint_subprocess_shell(user_input):
    """Command injection via subprocess shell=True."""
    import subprocess
    subprocess.run("echo " + user_input, shell=True)

def taint_eval_input(user_input):
    """Code injection via eval."""
    result = eval(user_input)
    return result

def taint_open_path(user_path):
    """Path traversal via open."""
    with open(user_path) as f:
        return f.read()

def taint_yaml_load(user_data):
    """Unsafe YAML deserialization."""
    import yaml
    return yaml.load(user_data)

def taint_pickle_loads(user_data):
    """Unsafe pickle deserialization."""
    import pickle
    return pickle.loads(user_data)

def taint_exec_string(user_code):
    """Code injection via exec."""
    exec(user_code)

def taint_template_render(user_name, template_str):
    """XSS via unescaped rendering."""
    html = f"<h1>Hello {user_name}</h1>"
    return html


# --- TAINT_FLOW safe functions ---

def safe_taint_parameterized(cursor, user_input):
    """Parameterized SQL query."""
    cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))

def safe_taint_sanitize_int(user_input):
    """int() sanitization."""
    safe_id = int(user_input)
    return f"SELECT * FROM users WHERE id = {safe_id}"

def safe_taint_allowlist(user_input, allowed):
    """Allowlist validation."""
    if user_input in allowed:
        return f"Action: {user_input}"
    return "Invalid action"

def safe_taint_escape(user_input):
    """html.escape before rendering."""
    import html
    safe = html.escape(user_input)
    return f"<p>{safe}</p>"

def safe_taint_subprocess_list(user_input):
    """Subprocess with list args (no shell)."""
    import subprocess
    subprocess.run(["echo", user_input])
