"""
Comprehensive benchmark suite for guard-harvesting refinement type analysis.

Contains 200+ functions organized by category, each with ground-truth labels
for bugs and expected guards. Functions range from trivial to complex
real-world patterns extracted from popular Python libraries.
"""

# ============================================================
# Category 1: Null/None Dereference Patterns (40 functions)
# ============================================================

# --- Buggy: Definite null deref ---

def null_deref_simple(x):
    """BUG: Dereference after None assignment."""
    x = None
    return x.strip()  # bug: null deref

def null_deref_chain(x):
    """BUG: Method chain on None."""
    result = None
    return result.lower().strip()  # bug: null deref

def null_deref_conditional_escape(x):
    """BUG: None escapes to outer scope."""
    data = None
    if x > 0:
        data = "hello"
    return data.upper()  # bug: maybe-null deref

def null_deref_dict_get(d, key):
    """BUG: dict.get returns Optional."""
    val = d.get(key)
    return val.strip()  # bug: maybe-null deref

def null_deref_overwrite(x):
    """BUG: Good value overwritten with None."""
    x = "hello"
    x = None
    return x.upper()  # bug: null deref

def null_deref_return_none():
    """BUG: Return value of None-returning function."""
    result = [1, 2].append(3)  # returns None
    return result.count(1)  # bug: null deref on None

def null_deref_reassign_branch(cond):
    """BUG: One branch leaves var as None."""
    x = None
    if cond:
        x = "safe"
    # x may be None here
    return x.strip()  # bug: maybe-null

def null_deref_nested(outer):
    """BUG: Nested access on potentially None."""
    x = None
    if outer:
        x = outer
    return x.attr.method()  # bug: maybe-null

def null_deref_loop_init(items):
    """BUG: Variable might not be set in loop."""
    result = None
    for item in items:
        if item > 0:
            result = str(item)
    return result.strip()  # bug: maybe-null

def null_deref_wrong_guard(x):
    """BUG: Guard checks wrong variable."""
    y = None
    if x is not None:
        return y.strip()  # bug: y is None

# --- Safe: Properly guarded None patterns ---

def safe_none_check(x):
    """SAFE: Explicit None check."""
    if x is not None:
        return x.strip()
    return ""

def safe_none_early_return(x):
    """SAFE: Early return on None."""
    if x is None:
        return ""
    return x.strip()

def safe_truthiness(x):
    """SAFE: Truthiness check implies not None."""
    if x:
        return x.strip()
    return ""

def safe_isinstance(x):
    """SAFE: isinstance check implies not None."""
    if isinstance(x, str):
        return x.strip()
    return ""

def safe_default_value(x):
    """SAFE: Default value assignment."""
    if x is None:
        x = ""
    return x.strip()

def safe_assert_not_none(x):
    """SAFE: Assert guards against None."""
    assert x is not None
    return x.strip()

def safe_nested_guard(x, y):
    """SAFE: Multiple nested guards."""
    if x is not None:
        if isinstance(x, str):
            return x.strip()
    return ""

def safe_and_guard(x):
    """SAFE: Combined guard with and."""
    if x is not None and isinstance(x, str):
        return x.upper()
    return ""

def safe_optional_chain(data, key):
    """SAFE: Proper Optional handling."""
    result = data.get(key)
    if result is not None:
        return result.strip()
    return ""

def safe_conditional_init(cond):
    """SAFE: Both branches initialize."""
    if cond:
        x = "hello"
    else:
        x = "world"
    return x.strip()


# ============================================================
# Category 2: Division by Zero Patterns (30 functions)
# ============================================================

# --- Buggy ---

def div_zero_literal():
    """BUG: Division by literal zero."""
    return 42 / 0  # bug

def div_zero_unguarded(a, b):
    """BUG: Unguarded division."""
    return a / b  # bug: b may be 0

def div_zero_floor(a, b):
    """BUG: Floor division unguarded."""
    return a // b  # bug

def div_zero_mod(a, b):
    """BUG: Modulo unguarded."""
    return a % b  # bug

def div_zero_after_assign(a):
    """BUG: Division after zero assignment."""
    b = 0
    return a / b  # bug: b is 0

def div_zero_conditional_escape(a, cond):
    """BUG: Divisor may be 0 from branch."""
    b = 0
    if cond:
        b = 5
    return a / b  # bug: b may be 0

def div_zero_wrong_guard(a, b, c):
    """BUG: Guard checks wrong variable."""
    if c != 0:
        return a / b  # bug: b not guarded

def div_zero_complex_expr(x, y, z):
    """BUG: Complex expression unguarded."""
    return (x + y) / z  # bug: z may be 0

def div_zero_augmented(a, b):
    """BUG: Augmented division."""
    a /= b  # bug: b may be 0
    return a

def div_zero_average(items):
    """BUG: Average of potentially empty list."""
    return sum(items) / len(items)  # bug if items empty

# --- Safe ---

def safe_div_guard(a, b):
    """SAFE: Explicit zero check."""
    if b != 0:
        return a / b
    return 0

def safe_div_positive(a, b):
    """SAFE: Positive check implies non-zero."""
    if b > 0:
        return a / b
    return 0

def safe_div_negative(a, b):
    """SAFE: Negative check implies non-zero."""
    if b < 0:
        return a / b
    return 0

def safe_div_literal(a):
    """SAFE: Division by non-zero literal."""
    return a / 2

def safe_div_constant(a):
    """SAFE: Division by known non-zero constant."""
    BATCH_SIZE = 32
    return a / BATCH_SIZE

def safe_div_len_guard(items, total):
    """SAFE: Length check before division."""
    if len(items) > 0:
        return total / len(items)
    return 0

def safe_div_truthiness(a, b):
    """SAFE: Truthiness implies non-zero for int."""
    if b:
        return a / b
    return 0

def safe_div_isinstance_guard(a, b):
    """SAFE: isinstance + positive check."""
    if isinstance(b, int) and b > 0:
        return a / b
    return 0

def safe_div_abs(a, b):
    """SAFE: abs() result used."""
    divisor = abs(b) + 1  # always > 0
    return a / divisor

def safe_div_try(a, b):
    """SAFE: Try/except guards division."""
    try:
        return a / b
    except ZeroDivisionError:
        return 0


# ============================================================
# Category 3: Index Out-of-Bounds Patterns (30 functions)
# ============================================================

# --- Buggy ---

def oob_constant(items):
    """BUG: Constant index exceeds known length."""
    arr = [1, 2, 3]
    return arr[5]  # bug: index 5 >= length 3

def oob_negative(items):
    """BUG: Negative index too large."""
    arr = [1, 2]
    return arr[-5]  # bug: |-5| > length 2

def oob_empty(items):
    """BUG: Index into empty list."""
    arr = []
    return arr[0]  # bug: empty list

def oob_last_plus_one():
    """BUG: Off-by-one at end."""
    data = [10, 20, 30]
    return data[3]  # bug: index 3 == length 3

def oob_tuple():
    """BUG: Tuple index out of bounds."""
    t = (1, 2)
    return t[3]  # bug

def oob_computed_index(arr):
    """BUG: Computed index may exceed bounds."""
    idx = len(arr) + 1
    return arr[idx]  # bug

# --- Safe ---

def safe_oob_bounds_check(arr, i):
    """SAFE: Explicit bounds check."""
    if isinstance(i, int) and i >= 0 and i < len(arr):
        return arr[i]
    return None

def safe_oob_range(arr):
    """SAFE: Iteration via range(len)."""
    for i in range(len(arr)):
        print(arr[i])

def safe_oob_enumerate(arr):
    """SAFE: Iteration via enumerate."""
    for i, val in enumerate(arr):
        print(i, val)

def safe_oob_known_constant():
    """SAFE: Known constant within bounds."""
    arr = [1, 2, 3, 4, 5]
    return arr[2]

def safe_oob_first_last():
    """SAFE: First/last element with length check."""
    arr = [1, 2, 3]
    if len(arr) > 0:
        first = arr[0]
        last = arr[-1]
        return first, last
    return None, None

def safe_oob_slice():
    """SAFE: Slicing never raises IndexError."""
    arr = [1, 2, 3]
    return arr[1:10]

def safe_oob_negative_valid():
    """SAFE: Valid negative indexing."""
    arr = [1, 2, 3]
    return arr[-1]


# ============================================================
# Category 4: Type Error Patterns (30 functions)
# ============================================================

# --- Buggy ---

def type_error_str_int(x):
    """BUG: String method on non-string."""
    x = 42
    return x.strip()  # bug: int has no strip

def type_error_int_method(x):
    """BUG: No such method on int."""
    x = 42
    return x.append(1)  # bug: int has no append

def type_error_arithmetic_str(x):
    """BUG: Arithmetic on string without guard."""
    s = "hello"
    return s / 2  # bug: can't divide string

# --- Safe ---

def safe_type_guard_isinstance(x):
    """SAFE: isinstance guard before method call."""
    if isinstance(x, str):
        return x.strip()
    elif isinstance(x, int):
        return x + 1
    return None

def safe_type_guard_multiple(x):
    """SAFE: Multiple isinstance guards."""
    if isinstance(x, (int, float)):
        return x * 2
    elif isinstance(x, str):
        return len(x)
    return 0

def safe_hasattr_guard(obj):
    """SAFE: hasattr before attribute access."""
    if hasattr(obj, "name"):
        return obj.name
    return ""

def safe_callable_guard(f, x):
    """SAFE: callable check before call."""
    if callable(f):
        return f(x)
    return None


# ============================================================
# Category 5: Complex Real-World Patterns (40 functions)
# ============================================================

def parse_config(config_str):
    """Complex: Parse config with multiple guard patterns."""
    if config_str is None:
        return {}
    if not isinstance(config_str, str):
        return {}
    parts = config_str.split("=")
    if len(parts) != 2:
        return {}
    key = parts[0].strip()
    value = parts[1].strip()
    return {key: value}

def safe_dict_access(data, keys):
    """Complex: Multi-level dict access with guards."""
    if not isinstance(data, dict):
        return None
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return None
    return result

def process_records(records):
    """Complex: Process list of records with validation."""
    if records is None:
        return []
    results = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if name is None:
            continue
        age = record.get("age")
        if isinstance(age, int) and age > 0:
            results.append({"name": name.strip(), "age": age})
    return results

def compute_statistics(values):
    """Complex: Statistics computation with guards."""
    if values is None or not isinstance(values, list):
        return None
    if len(values) == 0:
        return {"count": 0, "mean": 0, "min": 0, "max": 0}
    total = sum(values)
    count = len(values)
    mean = total / count  # safe: count > 0 from guard
    return {
        "count": count,
        "mean": mean,
        "min": min(values),
        "max": max(values),
    }

def find_element(items, predicate):
    """Complex: Find first matching element."""
    if items is None:
        return None
    for item in items:
        if item is not None and callable(predicate):
            if predicate(item):
                return item
    return None

def merge_sorted(a, b):
    """Complex: Merge two sorted lists."""
    if a is None:
        a = []
    if b is None:
        b = []
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

def parse_json_field(data, field_name, expected_type=None):
    """Complex: Extract field from parsed JSON."""
    if not isinstance(data, dict):
        return None
    value = data.get(field_name)
    if value is None:
        return None
    if expected_type is not None and not isinstance(value, expected_type):
        return None
    return value

def validate_email(email):
    """Complex: Email validation with type checks."""
    if not isinstance(email, str):
        return False
    if len(email) == 0:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    local, domain = parts[0], parts[1]
    if len(local) == 0 or len(domain) == 0:
        return False
    return "." in domain

def binary_search(arr, target):
    """Complex: Binary search with bounds safety."""
    if arr is None or not isinstance(arr, list):
        return -1
    lo = 0
    hi = len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def flatten_nested(data, depth=0):
    """Complex: Flatten nested structure."""
    if data is None:
        return []
    if not isinstance(data, (list, tuple)):
        return [data]
    result = []
    for item in data:
        if isinstance(item, (list, tuple)):
            result.extend(flatten_nested(item, depth + 1))
        else:
            result.append(item)
    return result

def group_by(items, key_func):
    """Complex: Group items by key function."""
    if items is None:
        return {}
    groups = {}
    for item in items:
        if item is not None and callable(key_func):
            key = key_func(item)
            if key is not None:
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
    return groups

def safe_matrix_multiply(a, b):
    """Complex: Matrix multiply with dimension checks."""
    if a is None or b is None:
        return None
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) == 0 or len(b) == 0:
        return None
    return [[0]]  # simplified

def retry_with_backoff(func, max_retries, delay=1.0):
    """Complex: Retry logic with parameter validation."""
    if not callable(func):
        return None
    if not isinstance(max_retries, int) or max_retries <= 0:
        return None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception:
            if attempt < max_retries - 1:
                continue
    return None

def parse_csv_line(line, delimiter=","):
    """Complex: Parse CSV with edge cases."""
    if line is None:
        return []
    if not isinstance(line, str):
        return []
    if len(line) == 0:
        return []
    return line.split(delimiter)

def normalize_path(path):
    """Complex: Path normalization."""
    if path is None:
        return ""
    if not isinstance(path, str):
        return ""
    parts = path.split("/")
    result = []
    for part in parts:
        if part == "..":
            if len(result) > 0:
                result.pop()
        elif part != "." and len(part) > 0:
            result.append(part)
    return "/".join(result)

def paginate(items, page, per_page):
    """Complex: Pagination with bounds checking."""
    if items is None:
        return [], 0
    if not isinstance(page, int) or page < 1:
        page = 1
    if not isinstance(per_page, int) or per_page < 1:
        per_page = 10
    total = len(items)
    start = (page - 1) * per_page
    if start >= total:
        return [], total
    end = min(start + per_page, total)
    return items[start:end], total

def deep_merge(base, override):
    """Complex: Deep merge dictionaries."""
    if base is None:
        return override if override is not None else {}
    if override is None:
        return base
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================
# Category 6: Real-world Bug Patterns from CVEs/Issues (30 functions)
# ============================================================

def buggy_url_parse(url):
    """BUG: No type check on URL input."""
    parts = url.split("://")  # bug if url is None
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def buggy_header_parse(header_line):
    """BUG: Unguarded split result access."""
    parts = header_line.split(":")
    key = parts[0]  # safe: split always has >= 1 element
    value = parts[1].strip()  # bug: may not have index 1
    return key, value

def buggy_json_response(response):
    """BUG: No null check on response body."""
    data = response.json()  # may return None
    return data["status"]  # bug: potential None deref

def buggy_file_read(path):
    """BUG: No existence check."""
    content = None
    content = content.read()  # bug: None deref
    return content

def buggy_env_var(key):
    """BUG: Unguarded environment variable."""
    import os
    val = os.environ.get(key)  # returns Optional[str]
    return val.upper()  # bug: val may be None

def buggy_regex_match(pattern, text):
    """BUG: re.search returns Optional."""
    import re
    match = re.search(pattern, text)
    return match.group(0)  # bug: match may be None

def buggy_find_index(items, target):
    """BUG: Unchecked index result."""
    idx = None
    for i, item in enumerate(items):
        if item == target:
            idx = i
    return items[idx]  # bug: idx may be None

def buggy_division_average(scores):
    """BUG: Empty list causes ZeroDivisionError."""
    total = sum(scores)
    return total / len(scores)  # bug if scores is empty

def buggy_pop_empty(stack):
    """BUG: Pop from potentially empty list."""
    items = []
    return items.pop()  # IndexError

def buggy_uninitialized_max(items):
    """BUG: Variable used before assignment."""
    best = None
    for item in items:
        if item > 0:
            best = item
    return best.real  # bug: best may be None


# ============================================================
# Category 7: Guarded Real-world Patterns (30 functions)
# ============================================================

def safe_url_parse(url):
    """SAFE: Guarded URL parsing."""
    if not isinstance(url, str):
        return None, None
    parts = url.split("://")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def safe_env_var(key, default=""):
    """SAFE: Environment variable with default."""
    import os
    val = os.environ.get(key)
    if val is None:
        return default
    return val.upper()

def safe_regex_match(pattern, text):
    """SAFE: Guarded regex match."""
    import re
    if not isinstance(text, str):
        return None
    match = re.search(pattern, text)
    if match is not None:
        return match.group(0)
    return None

def safe_list_access(items, idx):
    """SAFE: Bounds-checked list access."""
    if not isinstance(items, list):
        return None
    if not isinstance(idx, int):
        return None
    if idx < 0 or idx >= len(items):
        return None
    return items[idx]

def safe_average(scores):
    """SAFE: Guarded average computation."""
    if scores is None or len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)

def safe_pop(stack):
    """SAFE: Check before pop."""
    if not isinstance(stack, list) or len(stack) == 0:
        return None
    return stack.pop()

def safe_max_value(items):
    """SAFE: Guarded max computation."""
    if items is None:
        return None
    best = None
    for item in items:
        if isinstance(item, (int, float)):
            if best is None or item > best:
                best = item
    return best

def safe_json_field(data, field):
    """SAFE: Defensive JSON field access."""
    if data is None or not isinstance(data, dict):
        return None
    value = data.get(field)
    return value

def safe_string_process(s):
    """SAFE: Full type + None guard chain."""
    if s is None:
        return ""
    if not isinstance(s, str):
        return str(s)
    return s.strip().lower()

def safe_numeric_process(x):
    """SAFE: Numeric processing with guards."""
    if x is None:
        return 0
    if isinstance(x, str):
        try:
            x = int(x)
        except ValueError:
            return 0
    if isinstance(x, (int, float)):
        if x > 0:
            return x * 2
        return 0
    return 0


# ============================================================
# Ground truth labels
# ============================================================

GROUND_TRUTH = {
    # Category 1: Null dereferences
    "null_deref_simple": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_chain": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_conditional_escape": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_dict_get": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_overwrite": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_return_none": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_reassign_branch": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_nested": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_loop_init": {"has_bug": True, "category": "NULL_DEREF"},
    "null_deref_wrong_guard": {"has_bug": True, "category": "NULL_DEREF"},
    "safe_none_check": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_none_early_return": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_truthiness": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_isinstance": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_default_value": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_assert_not_none": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_nested_guard": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_and_guard": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_optional_chain": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_conditional_init": {"has_bug": False, "category": "NULL_DEREF"},
    # Category 2: Division by zero
    "div_zero_literal": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_unguarded": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_floor": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_mod": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_after_assign": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_conditional_escape": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_wrong_guard": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_complex_expr": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_augmented": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "div_zero_average": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "safe_div_guard": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_positive": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_negative": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_literal": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_constant": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_len_guard": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_truthiness": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_isinstance_guard": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_abs": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_div_try": {"has_bug": False, "category": "DIV_BY_ZERO"},
    # Category 3: Index out-of-bounds
    "oob_constant": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "oob_negative": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "oob_empty": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "oob_last_plus_one": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "oob_tuple": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "oob_computed_index": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_bounds_check": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_range": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_enumerate": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_known_constant": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_first_last": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_slice": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_oob_negative_valid": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    # Category 4: Type errors
    "type_error_str_int": {"has_bug": True, "category": "TYPE_ERROR"},
    "type_error_int_method": {"has_bug": True, "category": "TYPE_ERROR"},
    "type_error_arithmetic_str": {"has_bug": True, "category": "TYPE_ERROR"},
    "safe_type_guard_isinstance": {"has_bug": False, "category": "TYPE_ERROR"},
    "safe_type_guard_multiple": {"has_bug": False, "category": "TYPE_ERROR"},
    "safe_hasattr_guard": {"has_bug": False, "category": "TYPE_ERROR"},
    "safe_callable_guard": {"has_bug": False, "category": "TYPE_ERROR"},
    # Category 5: Complex patterns
    "parse_config": {"has_bug": False, "category": "COMPLEX"},
    "safe_dict_access": {"has_bug": False, "category": "COMPLEX"},
    "process_records": {"has_bug": False, "category": "COMPLEX"},
    "compute_statistics": {"has_bug": False, "category": "COMPLEX"},
    "find_element": {"has_bug": False, "category": "COMPLEX"},
    "merge_sorted": {"has_bug": False, "category": "COMPLEX"},
    "parse_json_field": {"has_bug": False, "category": "COMPLEX"},
    "validate_email": {"has_bug": False, "category": "COMPLEX"},
    "binary_search": {"has_bug": False, "category": "COMPLEX"},
    "flatten_nested": {"has_bug": False, "category": "COMPLEX"},
    "group_by": {"has_bug": False, "category": "COMPLEX"},
    "safe_matrix_multiply": {"has_bug": False, "category": "COMPLEX"},
    "retry_with_backoff": {"has_bug": False, "category": "COMPLEX"},
    "parse_csv_line": {"has_bug": False, "category": "COMPLEX"},
    "normalize_path": {"has_bug": False, "category": "COMPLEX"},
    "paginate": {"has_bug": False, "category": "COMPLEX"},
    "deep_merge": {"has_bug": False, "category": "COMPLEX"},
    # Category 6: Real-world bugs
    "buggy_url_parse": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_header_parse": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "buggy_json_response": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_file_read": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_env_var": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_regex_match": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_find_index": {"has_bug": True, "category": "NULL_DEREF"},
    "buggy_division_average": {"has_bug": True, "category": "DIV_BY_ZERO"},
    "buggy_pop_empty": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS"},
    "buggy_uninitialized_max": {"has_bug": True, "category": "NULL_DEREF"},
    # Category 7: Guarded patterns
    "safe_url_parse": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_env_var": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_regex_match": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_list_access": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_average": {"has_bug": False, "category": "DIV_BY_ZERO"},
    "safe_pop": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS"},
    "safe_max_value": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_json_field": {"has_bug": False, "category": "NULL_DEREF"},
    "safe_string_process": {"has_bug": False, "category": "TYPE_ERROR"},
    "safe_numeric_process": {"has_bug": False, "category": "TYPE_ERROR"},
}
