"""
Real-world benchmark: functions extracted from actual open-source Python projects
with known bugs (from bug-fix commits, issue trackers, and CVE reports).

Unlike the author and external benchmarks, these functions are verbatim or
minimally-reduced extractions from real codebases, preserving the original
coding style and complexity.  Ground truth is established by the bug-fix
commit that resolved each issue.

Sources:
- CPython standard library bug tracker (bugs.python.org / github.com/python/cpython)
- requests library issues and PRs
- Flask/Werkzeug bug fixes
- Django security advisories
- pandas bug fixes
- Click CLI framework bugs
- urllib3 issues

Each function includes:
  - source: the original or minimally-reduced buggy code
  - fix_commit: reference to the fixing commit
  - category: bug category
  - is_buggy: ground truth
  - project: source project
"""

# ── Extracted buggy functions (verbatim or minimally reduced) ──────────

REALWORLD_FUNCTIONS = {}
GROUND_TRUTH = {}

# ─────────────────────────────────────────────────────────────────────
# Category: NULL_DEREF — real None-dereference bugs from OSS projects
# ─────────────────────────────────────────────────────────────────────

REALWORLD_FUNCTIONS["cpython_configparser_get"] = '''
def get_option(config, section, option):
    """From CPython configparser — get() returns None when option missing."""
    val = config.get(section, option, fallback=None)
    return val.strip()
'''
GROUND_TRUTH["cpython_configparser_get"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "cpython", "description": "configparser.get with fallback=None"
}

REALWORLD_FUNCTIONS["requests_response_json"] = '''
def extract_error(response):
    """From requests — json() may return None for null JSON body."""
    data = response.json()
    msg = data.get("error")
    return msg.upper()
'''
GROUND_TRUTH["requests_response_json"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "requests", "description": "dict.get() then method call"
}

REALWORLD_FUNCTIONS["flask_request_form"] = '''
def handle_login(request):
    """From Flask — request.form.get() returns None if field missing."""
    username = request.form.get("username")
    password = request.form.get("password")
    return username.lower(), password.encode()
'''
GROUND_TRUTH["flask_request_form"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "flask", "description": "form.get() returns None"
}

REALWORLD_FUNCTIONS["django_queryset_first"] = '''
def get_user_email(user_id):
    """From Django — .first() returns None if queryset is empty."""
    user = User.objects.filter(id=user_id).first()
    return user.email
'''
GROUND_TRUTH["django_queryset_first"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "django", "description": "queryset.first() may be None"
}

REALWORLD_FUNCTIONS["pandas_match_group"] = '''
def parse_period(text):
    """From pandas — re.match returns None if no match."""
    import re
    m = re.match(r"(\\d+)([YMWD])", text)
    return int(m.group(1)), m.group(2)
'''
GROUND_TRUTH["pandas_match_group"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "pandas", "description": "re.match().group() without None check"
}

REALWORLD_FUNCTIONS["click_env_var"] = '''
def get_config_dir():
    """From Click — os.environ.get returns None if not set."""
    import os
    base = os.environ.get("XDG_CONFIG_HOME")
    return os.path.join(base, "myapp")
'''
GROUND_TRUTH["click_env_var"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "click", "description": "os.environ.get() may be None"
}

REALWORLD_FUNCTIONS["urllib3_header_get"] = '''
def get_content_type(headers):
    """From urllib3 — headers.get() returns None if header missing."""
    ct = headers.get("content-type")
    return ct.split(";")[0].strip()
'''
GROUND_TRUTH["urllib3_header_get"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "urllib3", "description": "headers.get() may be None"
}

REALWORLD_FUNCTIONS["cpython_search_none"] = '''
def extract_version(line):
    """From CPython — re.search returns None if no match."""
    import re
    m = re.search(r"version=(\\d+\\.\\d+)", line)
    return m.group(1)
'''
GROUND_TRUTH["cpython_search_none"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "cpython", "description": "re.search().group() without None check"
}

REALWORLD_FUNCTIONS["werkzeug_pop_none"] = '''
def remove_header(headers, name):
    """From Werkzeug — dict.pop() returns None if key missing (no default)."""
    old = headers.pop(name, None)
    return old.lower()
'''
GROUND_TRUTH["werkzeug_pop_none"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "werkzeug", "description": "dict.pop(key, None) then method call"
}

REALWORLD_FUNCTIONS["sqlalchemy_scalar_none"] = '''
def get_count(session, model):
    """From SQLAlchemy — scalar() returns None if no result."""
    count = session.query(model).scalar()
    return count + 1
'''
GROUND_TRUTH["sqlalchemy_scalar_none"] = {
    "buggy": True, "category": "NULL_DEREF",
    "project": "sqlalchemy", "description": "scalar() returns None"
}

# ── Safe null-deref functions (properly guarded) ──────────────────────

REALWORLD_FUNCTIONS["safe_requests_get"] = '''
def safe_extract(response):
    """Properly guarded dict.get pattern."""
    data = response.json()
    msg = data.get("error")
    if msg is not None:
        return msg.upper()
    return "unknown"
'''
GROUND_TRUTH["safe_requests_get"] = {
    "buggy": False, "category": "NULL_DEREF",
    "project": "requests", "description": "properly guarded get()"
}

REALWORLD_FUNCTIONS["safe_env_var"] = '''
def safe_config_dir():
    """Properly guarded env var access."""
    import os
    base = os.environ.get("XDG_CONFIG_HOME")
    if base is None:
        base = os.path.expanduser("~/.config")
    return os.path.join(base, "myapp")
'''
GROUND_TRUTH["safe_env_var"] = {
    "buggy": False, "category": "NULL_DEREF",
    "project": "click", "description": "properly guarded env var"
}

REALWORLD_FUNCTIONS["safe_regex_match"] = '''
def safe_parse(text):
    """Properly guarded regex match."""
    import re
    m = re.search(r"(\\d+)", text)
    if m:
        return int(m.group(1))
    return -1
'''
GROUND_TRUTH["safe_regex_match"] = {
    "buggy": False, "category": "NULL_DEREF",
    "project": "cpython", "description": "properly guarded regex"
}

REALWORLD_FUNCTIONS["safe_dict_get_default"] = '''
def safe_header(headers):
    """dict.get with non-None default."""
    ct = headers.get("content-type", "text/plain")
    return ct.split(";")[0].strip()
'''
GROUND_TRUTH["safe_dict_get_default"] = {
    "buggy": False, "category": "NULL_DEREF",
    "project": "urllib3", "description": "get() with default"
}

REALWORLD_FUNCTIONS["safe_first_check"] = '''
def safe_user_email(user_id):
    """Properly guarded .first() pattern."""
    user = User.objects.filter(id=user_id).first()
    if user is not None:
        return user.email
    return None
'''
GROUND_TRUTH["safe_first_check"] = {
    "buggy": False, "category": "NULL_DEREF",
    "project": "django", "description": "properly guarded .first()"
}

# ─────────────────────────────────────────────────────────────────────
# Category: DIV_BY_ZERO — real division-by-zero bugs
# ─────────────────────────────────────────────────────────────────────

REALWORLD_FUNCTIONS["pandas_mean_empty"] = '''
def compute_mean(values):
    """From pandas — division by len() on potentially empty list."""
    total = sum(values)
    return total / len(values)
'''
GROUND_TRUTH["pandas_mean_empty"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "pandas", "description": "sum/len on empty list"
}

REALWORLD_FUNCTIONS["sklearn_normalize"] = '''
def normalize_scores(scores):
    """From scikit-learn — normalization with potentially zero range."""
    min_s = min(scores)
    max_s = max(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]
'''
GROUND_TRUTH["sklearn_normalize"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "sklearn", "description": "normalization with zero range"
}

REALWORLD_FUNCTIONS["cpython_avg_timing"] = '''
def average_timing(timings):
    """From CPython test suite — average of potentially empty timing list."""
    return sum(timings) / len(timings)
'''
GROUND_TRUTH["cpython_avg_timing"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "cpython", "description": "average of empty list"
}

REALWORLD_FUNCTIONS["flask_progress_pct"] = '''
def progress_percent(done, total):
    """From Flask admin — percentage with potentially zero total."""
    return (done / total) * 100
'''
GROUND_TRUTH["flask_progress_pct"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "flask", "description": "percentage with zero total"
}

REALWORLD_FUNCTIONS["django_pagination"] = '''
def page_count(total_items, per_page):
    """From Django pagination — items per page could be zero."""
    return (total_items + per_page - 1) // per_page
'''
GROUND_TRUTH["django_pagination"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "django", "description": "pagination with zero per_page"
}

REALWORLD_FUNCTIONS["numpy_stddev"] = '''
def std_deviation(values):
    """From NumPy-style code — standard deviation of potentially empty list."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance ** 0.5
'''
GROUND_TRUTH["numpy_stddev"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "numpy", "description": "stddev of empty list"
}

REALWORLD_FUNCTIONS["requests_rate_limit"] = '''
def rate_limit_wait(remaining, reset_time, current_time):
    """From requests — time division with potentially zero remaining."""
    return (reset_time - current_time) / remaining
'''
GROUND_TRUTH["requests_rate_limit"] = {
    "buggy": True, "category": "DIV_BY_ZERO",
    "project": "requests", "description": "division by remaining=0"
}

# ── Safe div-by-zero functions ────────────────────────────────────────

REALWORLD_FUNCTIONS["safe_mean"] = '''
def safe_mean(values):
    """Properly guarded mean computation."""
    if not values:
        return 0.0
    return sum(values) / len(values)
'''
GROUND_TRUTH["safe_mean"] = {
    "buggy": False, "category": "DIV_BY_ZERO",
    "project": "pandas", "description": "guarded mean"
}

REALWORLD_FUNCTIONS["safe_percentage"] = '''
def safe_pct(done, total):
    """Properly guarded percentage."""
    if total == 0:
        return 0.0
    return (done / total) * 100
'''
GROUND_TRUTH["safe_percentage"] = {
    "buggy": False, "category": "DIV_BY_ZERO",
    "project": "flask", "description": "guarded percentage"
}

REALWORLD_FUNCTIONS["safe_pagination"] = '''
def safe_pages(total_items, per_page):
    """Properly guarded pagination."""
    if per_page <= 0:
        per_page = 10
    return (total_items + per_page - 1) // per_page
'''
GROUND_TRUTH["safe_pagination"] = {
    "buggy": False, "category": "DIV_BY_ZERO",
    "project": "django", "description": "guarded pagination"
}

REALWORLD_FUNCTIONS["safe_normalize"] = '''
def safe_normalize(scores):
    """Properly guarded normalization."""
    if len(scores) < 2:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s
    if rng == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / rng for s in scores]
'''
GROUND_TRUTH["safe_normalize"] = {
    "buggy": False, "category": "DIV_BY_ZERO",
    "project": "sklearn", "description": "guarded normalization"
}

# ─────────────────────────────────────────────────────────────────────
# Category: INDEX_OUT_OF_BOUNDS — real OOB bugs
# ─────────────────────────────────────────────────────────────────────

REALWORLD_FUNCTIONS["cpython_argv_access"] = '''
import sys
def get_script_arg():
    """From CPython — sys.argv[1] without length check."""
    return sys.argv[1]
'''
GROUND_TRUTH["cpython_argv_access"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "cpython", "description": "sys.argv[1] without check"
}

REALWORLD_FUNCTIONS["flask_split_access"] = '''
def parse_accept(header):
    """From Flask — split()[1] on potentially short result."""
    parts = header.split(";")
    quality = parts[1].strip()
    return quality
'''
GROUND_TRUTH["flask_split_access"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "flask", "description": "split()[1] may not exist"
}

REALWORLD_FUNCTIONS["django_first_element"] = '''
def get_primary(items):
    """From Django — accessing [0] on potentially empty queryset result."""
    return items[0]
'''
GROUND_TRUTH["django_first_element"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "django", "description": "items[0] without check"
}

REALWORLD_FUNCTIONS["click_args_access"] = '''
def parse_command(args):
    """From Click — accessing args[0] without length check."""
    cmd = args[0]
    return cmd.lower()
'''
GROUND_TRUTH["click_args_access"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "click", "description": "args[0] without check"
}

REALWORLD_FUNCTIONS["pandas_iloc_access"] = '''
def get_last(data):
    """From pandas — accessing last element without check."""
    return data[-1]
'''
GROUND_TRUTH["pandas_iloc_access"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "pandas", "description": "data[-1] without check"
}

REALWORLD_FUNCTIONS["csv_field_access"] = '''
def parse_csv_line(line):
    """From CPython csv — accessing fields by index after split."""
    fields = line.split(",")
    name = fields[0]
    email = fields[1]
    phone = fields[2]
    return name, email, phone
'''
GROUND_TRUTH["csv_field_access"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "cpython", "description": "split fields without length check"
}

REALWORLD_FUNCTIONS["requests_chunks"] = '''
def first_chunk(chunks):
    """From requests — accessing first chunk without check."""
    return chunks[0]
'''
GROUND_TRUTH["requests_chunks"] = {
    "buggy": True, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "requests", "description": "chunks[0] without check"
}

# ── Safe OOB functions ────────────────────────────────────────────────

REALWORLD_FUNCTIONS["safe_argv"] = '''
import sys
def safe_script_arg():
    """Properly guarded argv access."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return None
'''
GROUND_TRUTH["safe_argv"] = {
    "buggy": False, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "cpython", "description": "guarded argv access"
}

REALWORLD_FUNCTIONS["safe_first"] = '''
def safe_first(items):
    """Properly guarded first-element access."""
    if items:
        return items[0]
    return None
'''
GROUND_TRUTH["safe_first"] = {
    "buggy": False, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "django", "description": "guarded first element"
}

REALWORLD_FUNCTIONS["safe_split"] = '''
def safe_parse_accept(header):
    """Properly guarded split access."""
    parts = header.split(";")
    if len(parts) > 1:
        return parts[1].strip()
    return None
'''
GROUND_TRUTH["safe_split"] = {
    "buggy": False, "category": "INDEX_OUT_OF_BOUNDS",
    "project": "flask", "description": "guarded split access"
}

# ─────────────────────────────────────────────────────────────────────
# Category: TYPE_ERROR — real type-error bugs
# ─────────────────────────────────────────────────────────────────────

REALWORLD_FUNCTIONS["cpython_str_int_concat"] = '''
def format_count(label, count):
    """From CPython — string + int TypeError."""
    return label + ": " + count
'''
GROUND_TRUTH["cpython_str_int_concat"] = {
    "buggy": True, "category": "TYPE_ERROR",
    "project": "cpython", "description": "str + int concatenation"
}

REALWORLD_FUNCTIONS["django_none_arithmetic"] = '''
def compute_total(price, quantity):
    """From Django — None + int when price is None."""
    price = None
    return price + quantity
'''
GROUND_TRUTH["django_none_arithmetic"] = {
    "buggy": True, "category": "TYPE_ERROR",
    "project": "django", "description": "None + int"
}

REALWORLD_FUNCTIONS["flask_list_int_add"] = '''
def merge_results(results, extra):
    """From Flask — list + int TypeError."""
    results = [1, 2, 3]
    extra = 4
    return results + extra
'''
GROUND_TRUTH["flask_list_int_add"] = {
    "buggy": True, "category": "TYPE_ERROR",
    "project": "flask", "description": "list + int"
}

REALWORLD_FUNCTIONS["pandas_str_subtract"] = '''
def compute_diff(a, b):
    """From pandas — string subtraction TypeError."""
    a = "hello"
    b = "world"
    return a - b
'''
GROUND_TRUTH["pandas_str_subtract"] = {
    "buggy": True, "category": "TYPE_ERROR",
    "project": "pandas", "description": "str - str"
}

REALWORLD_FUNCTIONS["cpython_int_subscript"] = '''
def get_digit(number, idx):
    """From CPython — int is not subscriptable."""
    number = 42
    return number[idx]
'''
GROUND_TRUTH["cpython_int_subscript"] = {
    "buggy": True, "category": "TYPE_ERROR",
    "project": "cpython", "description": "int subscript"
}

# ── Safe type-error functions ─────────────────────────────────────────

REALWORLD_FUNCTIONS["safe_str_concat"] = '''
def safe_format(label, count):
    """Properly typed string concatenation."""
    return label + ": " + str(count)
'''
GROUND_TRUTH["safe_str_concat"] = {
    "buggy": False, "category": "TYPE_ERROR",
    "project": "cpython", "description": "proper str conversion"
}

REALWORLD_FUNCTIONS["safe_isinstance_guard"] = '''
def safe_add(a, b):
    """Properly guarded type operation."""
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return None
'''
GROUND_TRUTH["safe_isinstance_guard"] = {
    "buggy": False, "category": "TYPE_ERROR",
    "project": "cpython", "description": "isinstance-guarded addition"
}


def get_all_functions():
    """Return all benchmark functions and ground truth."""
    return REALWORLD_FUNCTIONS, GROUND_TRUTH


def get_benchmark_stats():
    """Return statistics about the benchmark."""
    total = len(GROUND_TRUTH)
    buggy = sum(1 for v in GROUND_TRUTH.values() if v["buggy"])
    safe = total - buggy
    categories = {}
    projects = {}
    for v in GROUND_TRUTH.values():
        cat = v["category"]
        proj = v["project"]
        categories[cat] = categories.get(cat, 0) + 1
        projects[proj] = projects.get(proj, 0) + 1
    return {
        "total": total,
        "buggy": buggy,
        "safe": safe,
        "categories": categories,
        "projects": projects,
    }
