"""
Visualization utilities for diversity analysis.

Provides static matplotlib plots, interactive HTML explorers,
and full dashboards for inspecting response diversity.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy as np

from .embedding import TextEmbedder, embed_texts, reduce_dim
from .diversity_metrics import cosine_diversity


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------

def _reduce_2d(embeddings: np.ndarray, method: str = "tsne") -> np.ndarray:
    """Reduce embeddings to 2-D for plotting.

    Uses internal PCA from ``reduce_dim`` for the 'pca' method.
    For 'tsne', we implement a lightweight Barnes-Hut–style approximation
    using gradient descent on the KL divergence of pairwise affinities.
    """
    if method == "pca":
        return reduce_dim(embeddings, target_dim=2, method="pca")

    # Lightweight t-SNE implementation
    n = embeddings.shape[0]
    if n <= 2:
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else embeddings

    perplexity = min(30.0, max(1.0, n / 4.0))
    # Pairwise squared distances
    sq = np.sum(embeddings ** 2, axis=1, keepdims=True)
    D = sq + sq.T - 2.0 * embeddings @ embeddings.T
    D = np.maximum(D, 0.0)

    # Compute joint probabilities
    P = np.exp(-D / (2.0 * perplexity))
    np.fill_diagonal(P, 0.0)
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)

    Y = np.random.RandomState(42).randn(n, 2) * 0.01
    lr = 100.0

    for iteration in range(300):
        sq_y = np.sum(Y ** 2, axis=1, keepdims=True)
        D_y = sq_y + sq_y.T - 2.0 * Y @ Y.T
        D_y = np.maximum(D_y, 0.0)
        Q = 1.0 / (1.0 + D_y)
        np.fill_diagonal(Q, 0.0)
        Q_sum = Q.sum()
        Q_norm = Q / max(Q_sum, 1e-12)
        Q_norm = np.maximum(Q_norm, 1e-12)

        PQ = P - Q_norm
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4.0 * (PQ[i] * Q[i])[:, None].T @ diff

        Y -= lr * grad
        Y -= Y.mean(axis=0)

    return Y


# ---------------------------------------------------------------------------
# Matplotlib-based plots
# ---------------------------------------------------------------------------

def plot_response_space(
    responses: List[str],
    method: str = "tsne",
    labels: Optional[List[str]] = None,
    title: str = "Response Space",
    embedder: Optional[TextEmbedder] = None,
) -> Any:
    """2-D scatter plot of response embeddings.

    Returns a matplotlib ``Figure``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if embedder is None:
        embedder = TextEmbedder(dim=64)
    embs = embedder.embed_batch(responses)
    coords = _reduce_2d(embs, method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=60, alpha=0.7, edgecolors="k", linewidths=0.5)

    if labels:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    div_score = cosine_diversity(embs)
    ax.set_title(f"{title}  (diversity = {div_score:.3f})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    return fig


def plot_coverage_map(
    responses: List[str],
    universe_samples: List[str],
    radius: float = 0.3,
    embedder: Optional[TextEmbedder] = None,
) -> Any:
    """Visualize which parts of the universe are covered by responses.

    Covered universe points are green; uncovered are red.  Response points
    are blue stars.  Returns a matplotlib ``Figure``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if embedder is None:
        embedder = TextEmbedder(dim=64)

    resp_embs = embedder.embed_batch(responses)
    uni_embs = embedder.embed_batch(universe_samples)
    all_embs = np.vstack([resp_embs, uni_embs])
    coords = _reduce_2d(all_embs, method="pca")

    resp_coords = coords[: len(responses)]
    uni_coords = coords[len(responses) :]

    # Determine coverage using pairwise distances in embedding space
    norms_r = np.linalg.norm(resp_embs, axis=1, keepdims=True)
    norms_u = np.linalg.norm(uni_embs, axis=1, keepdims=True)
    normed_r = resp_embs / np.maximum(norms_r, 1e-12)
    normed_u = uni_embs / np.maximum(norms_u, 1e-12)
    sim = normed_u @ normed_r.T  # (n_uni, n_resp)
    covered = sim.max(axis=1) >= (1.0 - radius)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        uni_coords[covered, 0], uni_coords[covered, 1],
        c="green", alpha=0.4, s=30, label="Covered",
    )
    ax.scatter(
        uni_coords[~covered, 0], uni_coords[~covered, 1],
        c="red", alpha=0.4, s=30, label="Uncovered",
    )
    ax.scatter(
        resp_coords[:, 0], resp_coords[:, 1],
        c="blue", marker="*", s=150, zorder=5, label="Responses",
    )

    frac = covered.mean()
    ax.set_title(f"Coverage Map  ({frac:.0%} covered, radius={radius})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_diversity_over_time(
    history: List[List[str]],
    embedder: Optional[TextEmbedder] = None,
) -> Any:
    """Line plot of diversity score across successive response sets.

    Parameters
    ----------
    history : list of list of str
        Each inner list is a set of responses at one time step.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if embedder is None:
        embedder = TextEmbedder(dim=64)

    scores = []
    for responses in history:
        if len(responses) < 2:
            scores.append(0.0)
        else:
            embs = embedder.embed_batch(responses)
            scores.append(float(cosine_diversity(embs)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(scores)), scores, marker="o", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Diversity Score")
    ax.set_title("Diversity Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive HTML explorer
# ---------------------------------------------------------------------------

def interactive_explorer(
    responses: List[str],
    method: str = "tsne",
    embedder: Optional[TextEmbedder] = None,
) -> str:
    """Generate a self-contained HTML page for interactively exploring responses.

    Each point can be hovered to see the full response text.

    Returns
    -------
    str
        HTML string.
    """
    if embedder is None:
        embedder = TextEmbedder(dim=64)
    embs = embedder.embed_batch(responses)
    coords = _reduce_2d(embs, method=method)
    div_score = cosine_diversity(embs)

    points = [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]),
         "text": responses[i][:200].replace('"', '\\"').replace("\n", " ")}
        for i in range(len(responses))
    ]
    data_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Diversity Explorer</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
#canvas {{ border: 1px solid #ccc; cursor: crosshair; }}
#tooltip {{ position: absolute; background: #333; color: #fff;
  padding: 6px 10px; border-radius: 4px; font-size: 12px;
  max-width: 300px; pointer-events: none; display: none; }}
h2 {{ margin-bottom: 4px; }}
</style></head><body>
<h2>Response Diversity Explorer</h2>
<p>Diversity score: <b>{div_score:.4f}</b> &mdash; {len(responses)} responses</p>
<canvas id="canvas" width="700" height="500"></canvas>
<div id="tooltip"></div>
<script>
const data = {data_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const W = canvas.width, H = canvas.height, pad = 40;
let xs = data.map(d => d.x), ys = data.map(d => d.y);
let xmin = Math.min(...xs), xmax = Math.max(...xs);
let ymin = Math.min(...ys), ymax = Math.max(...ys);
let xr = xmax - xmin || 1, yr = ymax - ymin || 1;
function tx(v) {{ return pad + (v - xmin) / xr * (W - 2*pad); }}
function ty(v) {{ return pad + (v - ymin) / yr * (H - 2*pad); }}
function draw() {{
  ctx.clearRect(0, 0, W, H);
  data.forEach(d => {{
    ctx.beginPath();
    ctx.arc(tx(d.x), ty(d.y), 6, 0, 2*Math.PI);
    ctx.fillStyle = '#4A90D9'; ctx.fill();
    ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.stroke();
  }});
}}
draw();
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let hit = null;
  data.forEach(d => {{
    if (Math.hypot(tx(d.x) - mx, ty(d.y) - my) < 10) hit = d;
  }});
  if (hit) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.pageX + 12) + 'px';
    tooltip.style.top = (e.pageY + 12) + 'px';
    tooltip.textContent = hit.text;
  }} else {{ tooltip.style.display = 'none'; }}
}});
</script></body></html>"""
    return html


# ---------------------------------------------------------------------------
# Full HTML dashboard
# ---------------------------------------------------------------------------

def diversity_dashboard(
    results: Dict[str, Any],
    embedder: Optional[TextEmbedder] = None,
) -> str:
    """Generate a comprehensive HTML dashboard for diversity analysis.

    Parameters
    ----------
    results : dict
        Expected keys (all optional):
        - ``'responses'``: list of str
        - ``'history'``: list of list of str (diversity over time)
        - ``'scores'``: dict of metric name → float
        - ``'metadata'``: dict of key → value

    Returns
    -------
    str
        Self-contained HTML string.
    """
    if embedder is None:
        embedder = TextEmbedder(dim=64)

    responses = results.get("responses", [])
    history = results.get("history", [])
    scores = results.get("scores", {})
    metadata = results.get("metadata", {})

    # Compute scores if not provided
    if responses and not scores:
        embs = embedder.embed_batch(responses)
        scores["cosine_diversity"] = float(cosine_diversity(embs))
        scores["n_responses"] = len(responses)

    # Build scatter data
    scatter_json = "[]"
    if responses:
        embs = embedder.embed_batch(responses)
        coords = _reduce_2d(embs, method="pca")
        pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                "t": responses[i][:120].replace('"', '\\"').replace("\n", " ")}
               for i in range(len(responses))]
        scatter_json = json.dumps(pts)

    # Build time-series data
    ts_json = "[]"
    if history:
        ts_vals = []
        for h in history:
            if len(h) < 2:
                ts_vals.append(0.0)
            else:
                ts_vals.append(float(cosine_diversity(embedder.embed_batch(h))))
        ts_json = json.dumps(ts_vals)

    scores_html = "".join(
        f"<tr><td>{k}</td><td><b>{v:.4f}</b></td></tr>"
        if isinstance(v, float) else f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in scores.items()
    )
    meta_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metadata.items()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Diversity Dashboard</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; }}
.header {{ background: #2c3e50; color: #fff; padding: 16px 24px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }}
.card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  padding: 16px; }}
.card h3 {{ margin-top: 0; color: #2c3e50; }}
table {{ width: 100%; border-collapse: collapse; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
canvas {{ width: 100%; height: 280px; }}
#tooltip2 {{ position: absolute; background: #333; color: #fff;
  padding: 4px 8px; border-radius: 3px; font-size: 11px;
  max-width: 250px; display: none; pointer-events: none; }}
</style></head><body>
<div class="header"><h1>Diversity Dashboard</h1></div>
<div class="grid">
<div class="card"><h3>Metrics</h3><table>{scores_html}</table></div>
<div class="card"><h3>Metadata</h3><table>{meta_html if meta_html else '<tr><td>—</td></tr>'}</table></div>
<div class="card"><h3>Response Space (PCA)</h3>
<canvas id="scatter" width="400" height="280"></canvas>
<div id="tooltip2"></div></div>
<div class="card"><h3>Diversity Over Time</h3>
<canvas id="timeseries" width="400" height="280"></canvas></div>
</div>
<script>
// Scatter
(function() {{
  const data = {scatter_json};
  const c = document.getElementById('scatter');
  const ctx = c.getContext('2d');
  if (!data.length) return;
  const W = c.width, H = c.height, p = 30;
  let xs = data.map(d=>d.x), ys = data.map(d=>d.y);
  let xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys);
  let xr=xx-xn||1, yr=yx-yn||1;
  data.forEach(d => {{
    ctx.beginPath();
    ctx.arc(p+(d.x-xn)/xr*(W-2*p), p+(d.y-yn)/yr*(H-2*p), 5, 0, 2*Math.PI);
    ctx.fillStyle='#3498db'; ctx.fill();
  }});
  const tt = document.getElementById('tooltip2');
  c.addEventListener('mousemove', e => {{
    const r = c.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    let hit=null;
    data.forEach(d => {{
      let px=p+(d.x-xn)/xr*(W-2*p), py=p+(d.y-yn)/yr*(H-2*p);
      if(Math.hypot(px-mx,py-my)<8) hit=d;
    }});
    if(hit) {{ tt.style.display='block'; tt.style.left=(e.pageX+10)+'px';
      tt.style.top=(e.pageY+10)+'px'; tt.textContent=hit.t; }}
    else {{ tt.style.display='none'; }}
  }});
}})();
// Time series
(function() {{
  const vals = {ts_json};
  const c = document.getElementById('timeseries');
  const ctx = c.getContext('2d');
  if (!vals.length) return;
  const W = c.width, H = c.height, p = 30;
  const mx = Math.max(...vals) || 1;
  ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 2;
  ctx.beginPath();
  vals.forEach((v,i) => {{
    let x = p + i/(vals.length-1||1)*(W-2*p);
    let y = H - p - (v/mx)*(H-2*p);
    i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  }});
  ctx.stroke();
  vals.forEach((v,i) => {{
    let x = p + i/(vals.length-1||1)*(W-2*p);
    let y = H - p - (v/mx)*(H-2*p);
    ctx.beginPath(); ctx.arc(x,y,3,0,2*Math.PI);
    ctx.fillStyle='#e74c3c'; ctx.fill();
  }});
}})();
</script></body></html>"""
    return html
