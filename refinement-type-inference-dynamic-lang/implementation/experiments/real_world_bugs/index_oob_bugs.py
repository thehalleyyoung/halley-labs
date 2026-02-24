# Real-world index out-of-bounds bugs from popular Python projects

# ── Bug 44: Empty list access (BugsInPy pandas-20) ──
def get_first_result(results):
    """Bug: accessing [0] on possibly empty list."""
    return results[0]  # BUG: results may be empty

# ── Bug 45: Hardcoded index (common pattern) ──
def parse_csv_line(line):
    """Bug: split may return fewer parts than expected."""
    parts = line.split(",")
    name = parts[0]
    age = parts[1]   # BUG: may not have 2 parts
    email = parts[2]  # BUG: may not have 3 parts
    return name, age, email

# ── Bug 46: Negative indexing assumption (tornado pattern) ──
def get_extension(filename):
    """Bug: filename without dot, split returns 1 element."""
    parts = filename.split(".")
    return parts[-1]  # Works but parts[1] would crash

# ── Bug 47: List comprehension index (django pattern) ──
def get_last_segment(path):
    """Bug: path could be empty string, split gives ['']."""
    segments = path.strip("/").split("/")
    return segments[-1]  # May return '' but not crash; segments[1] would

# ── Bug 48: Off-by-one in slice (common pattern) ──
def get_pairs(items):
    """Bug: accessing i+1 when i is last index."""
    pairs = []
    for i in range(len(items)):
        pairs.append((items[i], items[i + 1]))  # BUG: IndexError at last element
    return pairs

# ── Bug 49: Dictionary key error (treated as index OOB) ──
def lookup_status_code(code):
    """Bug: code may not be in the lookup table."""
    STATUS = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}
    return STATUS[code]  # BUG: KeyError if code not in STATUS

# ── Bug 50: Guarded index (TRUE NEGATIVE) ──
def safe_first(results):
    """Not a bug: properly guarded."""
    if results:
        return results[0]
    return None

# ── Bug 51: Length check (TRUE NEGATIVE) ──
def safe_csv_parse(line):
    """Not a bug: length checked before access."""
    parts = line.split(",")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None

# ── Bug 52: String indexing (scrapy pattern) ──
def extract_protocol(url):
    """Bug: url could be empty."""
    return url[0:5]  # Slice is safe, but url[0] would crash on empty

# ── Bug 53: argv access (cli pattern) ──
def parse_args(argv):
    """Bug: argv may have no arguments beyond script name."""
    command = argv[1]  # BUG: IndexError if len(argv) < 2
    return command

# ── Bug 54: Multi-dim indexing (numpy pattern) ──
def get_cell(matrix, row, col):
    """Bug: row or col may be out of bounds."""
    return matrix[row][col]  # BUG: IndexError possible

# ── Bug 55: Pop from empty (common pattern) ──
def process_stack(stack):
    """Bug: popping from empty list."""
    item = stack.pop()  # BUG: IndexError if stack is empty
    return item * 2
