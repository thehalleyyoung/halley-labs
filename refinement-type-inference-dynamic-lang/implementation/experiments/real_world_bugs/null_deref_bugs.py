# Real-world null dereference bugs from popular Python projects
# Each function represents a minimal reproduction of an actual bug pattern

# ── Bug 1: pandas DataFrame.get pattern (BugsInPy pandas-49) ──
def pandas_get_column(df, col_name):
    """Bug: df.get returns None if column missing, then .values crashes."""
    result = df.get(col_name)
    return result.values  # BUG: result may be None

# ── Bug 2: requests response handling (BugsInPy requests-2) ──
def requests_json_parse(response):
    """Bug: response.json() may return None for empty body."""
    data = response.json()
    return data["key"]  # BUG: data may be None

# ── Bug 3: flask request.args pattern (BugsInPy flask-2) ──
def flask_get_param(request):
    """Bug: request.args.get returns None by default."""
    page = request.args.get("page")
    offset = page * 10  # BUG: page is None, TypeError on multiply

# ── Bug 4: django queryset .first() (GitHub django/django#28519) ──
def django_first_user(User, username):
    """Bug: .first() returns None if no match."""
    user = User.objects.filter(username=username).first()
    return user.email  # BUG: user may be None

# ── Bug 5: dict.get without default (common pattern) ──
def config_lookup(config, section):
    """Bug: dict.get returns None, then subscript fails."""
    settings = config.get(section)
    return settings["timeout"]  # BUG: settings may be None

# ── Bug 6: re.match returns None (BugsInPy youtube-dl-15) ──
def parse_video_id(url):
    """Bug: re.match returns None if pattern doesn't match."""
    import re
    match = re.match(r"v=([a-zA-Z0-9]+)", url)
    return match.group(1)  # BUG: match may be None

# ── Bug 7: os.environ.get (common pattern) ──
def get_db_connection(os):
    """Bug: os.environ.get returns None if env var not set."""
    host = os.environ.get("DB_HOST")
    return host.split(":")[0]  # BUG: host may be None

# ── Bug 8: find() returns None (BugsInPy scrapy-1) ──
def extract_title(soup):
    """Bug: BeautifulSoup find returns None if element not found."""
    title_tag = soup.find("title")
    return title_tag.text  # BUG: title_tag may be None

# ── Bug 9: next() with StopIteration → caught as None (common) ──
def find_first_match(items, predicate):
    """Bug: next() with default=None, then attribute access."""
    result = next((x for x in items if predicate(x)), None)
    return result.name  # BUG: result may be None

# ── Bug 10: json.loads returning None (edge case) ──
def parse_api_response(text):
    """Bug: json.loads('null') returns None."""
    import json
    data = json.loads(text)
    return data["status"]  # BUG: data may be None if text is "null"

# ── Bug 11: Optional return not checked (BugsInPy black-1) ──
def find_project_root(srcs):
    """Bug: function returns None on failure, caller doesn't check."""
    root = _find_root(srcs)
    config = root / "pyproject.toml"  # BUG: root may be None
    return config

def _find_root(srcs):
    if not srcs:
        return None
    return srcs[0]

# ── Bug 12: pop from dict returns None (BugsInPy httpie-2) ──
def process_headers(headers):
    """Bug: dict.pop with default None, then method call."""
    content_type = headers.pop("Content-Type", None)
    return content_type.lower()  # BUG: content_type may be None

# ── Bug 13: getattr with None default ──
def get_model_name(model):
    """Bug: getattr returns None default, then string op."""
    name = getattr(model, "name", None)
    return name.replace("_", " ")  # BUG: name may be None

# ── Bug 14: Guarded access (TRUE NEGATIVE - should NOT flag) ──
def safe_get_column(df, col_name):
    """Not a bug: properly guarded with is not None."""
    result = df.get(col_name)
    if result is not None:
        return result.values
    return None

# ── Bug 15: Guarded with isinstance (TRUE NEGATIVE) ──
def safe_parse(data):
    """Not a bug: guarded with isinstance."""
    import json
    result = json.loads(data)
    if isinstance(result, dict):
        return result["key"]
    return None

# ── Bug 16: Early return guard (TRUE NEGATIVE) ──
def safe_match(url):
    """Not a bug: early return guards None."""
    import re
    match = re.match(r"v=([a-zA-Z0-9]+)", url)
    if match is None:
        return ""
    return match.group(1)

# ── Bug 17: Chained None propagation (BugsInPy tornado-2) ──
def tornado_get_cookie(request):
    """Bug: get_cookie returns None, then .decode() crashes."""
    cookie = request.cookies.get("session")
    token = cookie.value  # BUG: cookie may be None
    return token.decode("utf-8")

# ── Bug 18: SQLAlchemy query result (GitHub issue pattern) ──
def get_user_role(session, user_id):
    """Bug: .one_or_none() returns None."""
    user = session.query("User").filter_by(id=user_id).one_or_none()
    return user.role  # BUG: user may be None
