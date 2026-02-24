# Real-world type error bugs from popular Python projects

# ── Bug 19: String + int concatenation (BugsInPy tqdm-3) ──
def format_progress(current, total):
    """Bug: concatenating string with int without conversion."""
    msg = "Progress: " + current + "/" + total  # BUG: if current/total are int
    return msg

# ── Bug 20: Calling non-callable (common pattern) ──
def apply_transform(data, transform):
    """Bug: transform may be a string, not callable."""
    config = {"upper": "uppercase", "lower": "lowercase"}
    func = config.get(transform, transform)
    return func(data)  # BUG: func may be a string, not callable

# ── Bug 21: Iteration over non-iterable (BugsInPy luigi-3) ──
def process_dependencies(task):
    """Bug: requires() may return a single Task, not list."""
    deps = task.requires()
    for dep in deps:  # BUG: if deps is a single Task (not iterable list)
        dep.run()

# ── Bug 22: Integer used as string (common pattern) ──
def build_url(base, port, path):
    """Bug: port is int, used in string concatenation."""
    url = base + ":" + port + "/" + path  # BUG: port is int
    return url

# ── Bug 23: Wrong type to str.format (scikit-learn pattern) ──
def format_score(scores):
    """Bug: scores is a list, not a float."""
    msg = "Accuracy: {:.2f}".format(scores)  # BUG: scores is list
    return msg

# ── Bug 24: Bool treated as int in division ──
def compute_ratio(enabled, total):
    """Bug: enabled is bool but used in arithmetic that expects int."""
    if isinstance(enabled, bool):
        ratio = enabled / total  # Technically works but semantically wrong
    return ratio

# ── Bug 25: Guarded type check (TRUE NEGATIVE) ──
def safe_concat(a, b):
    """Not a bug: isinstance guard protects concatenation."""
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    return str(a) + str(b)

# ── Bug 26: Type narrowing after guard (TRUE NEGATIVE) ──
def process_input(value):
    """Not a bug: isinstance narrows type."""
    if isinstance(value, int):
        return value * 2
    elif isinstance(value, str):
        return value.upper()
    return str(value)

# ── Bug 27: Subscript on non-subscriptable (pandas pattern) ──
def get_series_value(series, idx):
    """Bug: series.name returns string, not subscriptable sometimes."""
    name = series.name
    prefix = name[0]  # May crash if name is None or int
    return prefix

# ── Bug 28: Method on wrong type (django ORM pattern) ──
def get_display_name(user):
    """Bug: user.get_full_name() returns str or None."""
    name = user.get_full_name()
    parts = name.split()  # BUG: name may be None
    return parts[0] if parts else ""

# ── Bug 29: Arithmetic on None (numpy pattern) ──
def normalize(values, scale):
    """Bug: scale could be None from failed computation."""
    factor = compute_scale(values)
    result = values / factor  # BUG: factor may be None
    return result

def compute_scale(values):
    if not values:
        return None
    return max(values)

# ── Bug 30: String method on bytes (requests pattern) ──
def decode_response(data):
    """Bug: data might be bytes, .upper() works differently."""
    if hasattr(data, "decode"):
        text = data  # still bytes!
    else:
        text = data
    return text.upper()  # Works but semantically wrong if bytes

# ── Bug 31: Guarded early return (TRUE NEGATIVE) ──
def safe_normalize(values):
    """Not a bug: early return on None."""
    factor = compute_scale(values)
    if factor is None:
        return values
    return [v / factor for v in values]
