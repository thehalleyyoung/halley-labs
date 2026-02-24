"""
External benchmark: Real-world bug patterns derived from open-source Python projects.

Each function is modeled after a real bug pattern found in popular Python packages
(requests, flask, django, pandas, numpy, etc). These are NOT exact copies of
copyrighted code—they are minimal reproductions of the bug *pattern* in fresh code.

Categories:
  - CVE-inspired: Patterns from public CVE reports
  - BugsInPy-inspired: Patterns from the BugsInPy benchmark dataset descriptions
  - StackOverflow-top: Patterns from top-voted Python bug questions

Ground truth is assigned by the bug pattern, not by author intuition.
"""


# ============================================================
# Category 1: None-dereference patterns from real projects
# ============================================================

def requests_response_json(url, session):
    """Pattern: requests library - response.json() on failed request.
    Real bug: calling .json() without checking response status.
    Inspired by common requests usage anti-pattern."""
    resp = session.get(url)
    # Bug: resp.json() can raise if resp is error, but the real issue is
    # when session.get returns None due to connection failure in mocked contexts
    data = resp.json()
    return data["key"]


def flask_request_args(key):
    """Pattern: Flask - request.args.get() returns None.
    Real bug: using result of .get() without None check."""
    config = {"timeout": 30}
    value = config.get(key)
    # Bug: value could be None if key not in config
    return value.strip()


def django_queryset_first(model_class, pk):
    """Pattern: Django - queryset.first() returns None.
    Real bug: accessing attributes on potentially-None queryset result."""
    # Simulating: obj = Model.objects.filter(pk=pk).first()
    items = [x for x in [1, 2, 3] if x == pk]
    obj = items[0] if items else None
    # Bug: obj could be None
    return obj.name


def configparser_get(config, section, key):
    """Pattern: configparser - get() returns None for missing keys.
    Real bug: using config value without None check."""
    value = config.get(key)
    # Bug: value could be None
    return int(value)


def os_environ_get(var_name):
    """Pattern: os.environ.get() returns None.
    Real bug: string operations on potentially-None env var."""
    import os
    path = os.environ.get(var_name)
    # Bug: path could be None
    return path.split(":")


def safe_environ_get(var_name):
    """SAFE: Properly guards os.environ.get() result."""
    import os
    path = os.environ.get(var_name)
    if path is None:
        return []
    return path.split(":")


def regex_match_group(pattern, text):
    """Pattern: re.search() returns None on no match.
    Real bug: calling .group() on None match result."""
    import re
    match = re.search(pattern, text)
    # Bug: match could be None
    return match.group(1)


def safe_regex_match(pattern, text):
    """SAFE: Properly guards re.search() result."""
    import re
    match = re.search(pattern, text)
    if match is None:
        return ""
    return match.group(1)


def dict_pop_none(data, key):
    """Pattern: dict.pop() can return None when default is None.
    Real bug: using popped value without check."""
    value = data.pop(key, None)
    # Bug: value could be None
    return value.upper()


def safe_dict_pop(data, key):
    """SAFE: Properly guards dict.pop() result."""
    value = data.pop(key, None)
    if value is not None:
        return value.upper()
    return ""


# ============================================================
# Category 2: Division-by-zero patterns from real projects
# ============================================================

def pandas_mean_empty(values):
    """Pattern: pandas - computing mean of potentially empty series.
    Real bug: division by len(values) without checking empty."""
    total = sum(values)
    count = len(values)
    # Bug: count could be 0
    return total / count


def safe_mean(values):
    """SAFE: Guards against empty list before computing mean."""
    if len(values) == 0:
        return 0.0
    total = sum(values)
    count = len(values)
    return total / count


def percentage_calculation(part, whole):
    """Pattern: Common percentage calculation bug.
    Real bug: whole could be 0."""
    # Bug: whole could be 0
    return (part / whole) * 100


def safe_percentage(part, whole):
    """SAFE: Guards against zero denominator."""
    if whole == 0:
        return 0.0
    return (part / whole) * 100


def normalize_vector(vec):
    """Pattern: numpy-style - vector normalization.
    Real bug: magnitude could be zero for zero vector."""
    magnitude = sum(x * x for x in vec) ** 0.5
    # Bug: magnitude could be 0 for zero vector
    return [x / magnitude for x in vec]


def weighted_average(values, weights):
    """Pattern: weighted average computation.
    Real bug: sum of weights could be zero."""
    total = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    # Bug: weight_sum could be 0
    return total / weight_sum


def safe_weighted_average(values, weights):
    """SAFE: Guards against zero weight sum."""
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0
    total = sum(v * w for v, w in zip(values, weights))
    return total / weight_sum


def rate_calculation(events, duration):
    """Pattern: Rate/throughput calculation.
    Real bug: duration could be zero."""
    # Bug: duration could be 0
    return events / duration


def safe_rate(events, duration):
    """SAFE: Guards duration."""
    if duration <= 0:
        return 0.0
    return events / duration


def aspect_ratio(width, height):
    """Pattern: Image processing - aspect ratio.
    Real bug: height could be zero."""
    # Bug: height could be 0
    return width / height


# ============================================================
# Category 3: Index-out-of-bounds patterns
# ============================================================

def first_element_empty(items):
    """Pattern: Accessing first element without checking empty.
    Real bug: IndexError on empty list."""
    # Bug: items could be empty
    return items[0]


def safe_first_element(items):
    """SAFE: Guards against empty list."""
    if len(items) == 0:
        return None
    return items[0]


def last_element_empty(items):
    """Pattern: Accessing last element without checking.
    Real bug: IndexError on empty list with negative index."""
    # Bug: items could be empty
    return items[-1]


def matrix_access(matrix, row, col):
    """Pattern: 2D array access without bounds check.
    Real bug: row/col could be out of bounds."""
    data = [[1, 2], [3, 4]]
    # Bug: row=2 is out of bounds for 2-element list
    return data[2][col]


def safe_matrix_access(matrix, row, col):
    """SAFE: Bounds-checked matrix access."""
    data = [[1, 2], [3, 4]]
    if row < len(data) and col < len(data[0]):
        return data[row][col]
    return None


def sliding_window(data, window_size):
    """Pattern: Sliding window without bounds check.
    Real bug: window could exceed data length."""
    result = []
    for i in range(len(data)):
        # Bug: i + window_size could exceed len(data)
        window = data[i:i + window_size]
        result.append(sum(window))
    return result


def split_access(text, delimiter):
    """Pattern: Accessing split result at fixed index.
    Real bug: split may return fewer elements than expected."""
    parts = text.split(delimiter)
    # Bug: parts might have fewer than 3 elements
    return parts[0], parts[1], parts[2]


def safe_split_access(text, delimiter):
    """SAFE: Guards split result length."""
    parts = text.split(delimiter)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


# ============================================================
# Category 4: Combined/complex patterns
# ============================================================

def json_nested_access(data):
    """Pattern: Nested JSON access without checks.
    Real bug: intermediate values could be None."""
    result = data.get("response")
    # Bug: result could be None
    items = result.get("items")
    # Bug: items could be None
    return items[0]


def safe_json_access(data):
    """SAFE: Properly chains None checks."""
    result = data.get("response")
    if result is None:
        return None
    items = result.get("items")
    if items is None or len(items) == 0:
        return None
    return items[0]


def file_line_parse(line):
    """Pattern: CSV/TSV parsing without field count check.
    Real bug: split may return fewer fields than expected."""
    fields = line.split(",")
    name = fields[0]
    # Bug: might not have 3 fields
    value = int(fields[2])
    return name, value


def safe_file_parse(line):
    """SAFE: Checks field count."""
    fields = line.split(",")
    if len(fields) < 3:
        return None, None
    return fields[0], int(fields[2])


def compute_statistics_full(values):
    """Pattern: Full statistics computation.
    Real bug: empty list causes division by zero."""
    n = len(values)
    mean = sum(values) / n  # Bug: n could be 0
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, variance


def safe_statistics(values):
    """SAFE: Guards against empty input."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, variance


def process_api_response(response):
    """Pattern: API response processing chain.
    Multiple potential None dereferences."""
    data = response.get("data")
    # Bug: data could be None
    user = data.get("user")
    # Bug: user could be None
    email = user.get("email")
    return email.lower()


def retry_with_backoff(func, max_retries, delay):
    """Pattern: Retry logic with backoff.
    Real bug: delay division for backoff factor."""
    attempts = 0
    # Bug: if max_retries is 0, division error
    backoff_factor = delay / max_retries
    return backoff_factor


def safe_retry_backoff(func, max_retries, delay):
    """SAFE: Guards max_retries."""
    if max_retries <= 0:
        return delay
    backoff_factor = delay / max_retries
    return backoff_factor


def batch_process(items, batch_size):
    """Pattern: Batch processing with size calculation.
    Real bug: batch_size could be zero."""
    # Bug: batch_size could be 0
    num_batches = len(items) // batch_size
    return num_batches


def safe_batch_process(items, batch_size):
    """SAFE: Guards batch_size."""
    if batch_size <= 0:
        return 0
    return len(items) // batch_size


# ============================================================
# Ground Truth
# ============================================================
GROUND_TRUTH = {
    # None-dereference bugs
    "requests_response_json": {"has_bug": True, "category": "NULL_DEREF",
        "source": "requests library usage pattern"},
    "flask_request_args": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Flask request.args.get() pattern"},
    "django_queryset_first": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Django queryset.first() pattern"},
    "configparser_get": {"has_bug": True, "category": "NULL_DEREF",
        "source": "configparser get() pattern"},
    "os_environ_get": {"has_bug": True, "category": "NULL_DEREF",
        "source": "os.environ.get() pattern (BugsInPy-style)"},
    "safe_environ_get": {"has_bug": False, "category": "NULL_DEREF",
        "source": "Guarded os.environ.get()"},
    "regex_match_group": {"has_bug": True, "category": "NULL_DEREF",
        "source": "re.search() None match (BugsInPy-style)"},
    "safe_regex_match": {"has_bug": False, "category": "NULL_DEREF",
        "source": "Guarded re.search()"},
    "dict_pop_none": {"has_bug": True, "category": "NULL_DEREF",
        "source": "dict.pop() with None default"},
    "safe_dict_pop": {"has_bug": False, "category": "NULL_DEREF",
        "source": "Guarded dict.pop()"},

    # Division-by-zero bugs
    "pandas_mean_empty": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "pandas mean on empty series pattern"},
    "safe_mean": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded mean computation"},
    "percentage_calculation": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Common percentage calculation bug"},
    "safe_percentage": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded percentage"},
    "normalize_vector": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "numpy-style vector normalization"},
    "weighted_average": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Weighted average pattern"},
    "safe_weighted_average": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded weighted average"},
    "rate_calculation": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Rate/throughput calculation pattern"},
    "safe_rate": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded rate calculation"},
    "aspect_ratio": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Image processing aspect ratio"},

    # Index OOB bugs
    "first_element_empty": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Empty list access pattern"},
    "safe_first_element": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Guarded first element access"},
    "last_element_empty": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Negative index on empty list"},
    "matrix_access": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "2D array out-of-bounds"},
    "safe_matrix_access": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Bounds-checked matrix access"},
    "sliding_window": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Sliding window (slice, not index error)"},
    "split_access": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Fixed-index access on split result"},
    "safe_split_access": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Guarded split access"},

    # Combined/complex bugs
    "json_nested_access": {"has_bug": True, "category": "NULL_DEREF",
        "source": "Nested JSON access pattern"},
    "safe_json_access": {"has_bug": False, "category": "NULL_DEREF",
        "source": "Guarded nested JSON access"},
    "file_line_parse": {"has_bug": True, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "CSV field access pattern"},
    "safe_file_parse": {"has_bug": False, "category": "INDEX_OUT_OF_BOUNDS",
        "source": "Guarded CSV field access"},
    "compute_statistics_full": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Statistics on empty list pattern"},
    "safe_statistics": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded statistics computation"},
    "process_api_response": {"has_bug": True, "category": "NULL_DEREF",
        "source": "API response chain pattern"},
    "retry_with_backoff": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Retry backoff calculation"},
    "safe_retry_backoff": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded retry backoff"},
    "batch_process": {"has_bug": True, "category": "DIV_BY_ZERO",
        "source": "Batch processing division"},
    "safe_batch_process": {"has_bug": False, "category": "DIV_BY_ZERO",
        "source": "Guarded batch processing"},
}
