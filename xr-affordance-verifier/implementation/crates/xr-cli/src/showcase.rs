//! Demo showcase bundle generation.
//!
//! Produces a before/after remediation package with scene JSON, certificates,
//! dashboards, SVG reports, a manifest, and a polished landing page.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use serde::Serialize;
use xr_lint::LintReport;
use xr_types::certificate::CoverageCertificate;
use xr_types::error::VerifierResult;
use xr_types::scene::SceneModel;

use crate::output::OutputFormatter;
use crate::pipeline::PipelineResult;
use crate::webapp::WebAppGenerator;
use crate::OutputFormat;

#[derive(Debug, Clone, Serialize)]
pub struct ShowcaseArtifactSet {
    pub bundle_manifest: String,
    pub landing_page: String,
    pub bundle_readme: String,
    pub command_script: String,
    pub before_scene: String,
    pub after_scene: String,
    pub before_certificate: String,
    pub after_certificate: String,
    pub before_dashboard: String,
    pub after_dashboard: String,
    pub before_svg: String,
    pub after_svg: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShowcaseSnapshot {
    pub scene_name: String,
    pub lint_errors: usize,
    pub lint_warnings: usize,
    pub total_findings: usize,
    pub grade: String,
    pub kappa: f64,
    pub pass_count: usize,
    pub sample_count: usize,
    pub verified_regions: usize,
    pub violations: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShowcaseDelta {
    pub kappa_delta: f64,
    pub violation_delta: isize,
    pub lint_error_delta: isize,
    pub lint_warning_delta: isize,
    pub improved_elements: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ElementImprovement {
    pub name: String,
    pub before_coverage: f64,
    pub after_coverage: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShowcaseManifest {
    pub scenario: String,
    pub title: String,
    pub summary: String,
    pub artifacts: ShowcaseArtifactSet,
    pub before: ShowcaseSnapshot,
    pub after: ShowcaseSnapshot,
    pub delta: ShowcaseDelta,
    pub remediation_notes: Vec<String>,
    pub lint_suggestions: Vec<String>,
    pub top_improvements: Vec<ElementImprovement>,
}

pub struct ShowcaseBundleGenerator;

pub struct ShowcaseInputs<'a> {
    pub title: &'a str,
    pub scenario: &'a str,
    pub before_scene: &'a SceneModel,
    pub after_scene: &'a SceneModel,
    pub before_lint: &'a LintReport,
    pub after_lint: &'a LintReport,
    pub before_pipeline: &'a PipelineResult,
    pub after_pipeline: &'a PipelineResult,
    pub before_certificate: &'a CoverageCertificate,
    pub after_certificate: &'a CoverageCertificate,
    pub remediation_notes: &'a [String],
}

impl ShowcaseBundleGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_bundle(
        &self,
        output_dir: &Path,
        inputs: ShowcaseInputs<'_>,
    ) -> VerifierResult<ShowcaseManifest> {
        std::fs::create_dir_all(output_dir)?;

        let formatter = OutputFormatter::new(OutputFormat::Text, false);
        let artifacts = ShowcaseArtifactSet {
            bundle_manifest: "showcase.bundle.json".into(),
            landing_page: "index.html".into(),
            bundle_readme: "README.md".into(),
            command_script: "demo_commands.sh".into(),
            before_scene: "before.scene.json".into(),
            after_scene: "after.scene.json".into(),
            before_certificate: "before.certificate.json".into(),
            after_certificate: "after.certificate.json".into(),
            before_dashboard: "before.dashboard.html".into(),
            after_dashboard: "after.dashboard.html".into(),
            before_svg: "before.report.svg".into(),
            after_svg: "after.report.svg".into(),
        };

        std::fs::write(
            output_dir.join(&artifacts.before_scene),
            xr_scene::parser::scene_to_json(inputs.before_scene)?,
        )?;
        std::fs::write(
            output_dir.join(&artifacts.after_scene),
            xr_scene::parser::scene_to_json(inputs.after_scene)?,
        )?;
        std::fs::write(
            output_dir.join(&artifacts.before_certificate),
            inputs.before_certificate.to_json()?,
        )?;
        std::fs::write(
            output_dir.join(&artifacts.after_certificate),
            inputs.after_certificate.to_json()?,
        )?;
        std::fs::write(
            output_dir.join(&artifacts.before_svg),
            formatter.generate_svg_report(inputs.before_certificate),
        )?;
        std::fs::write(
            output_dir.join(&artifacts.after_svg),
            formatter.generate_svg_report(inputs.after_certificate),
        )?;

        let before_dashboard = WebAppGenerator::new().generate(
            inputs.before_scene,
            Some(inputs.before_certificate),
            Some(inputs.before_pipeline),
            Some(&format!("{} — Before Remediation", inputs.title)),
        );
        let after_dashboard = WebAppGenerator::new().generate(
            inputs.after_scene,
            Some(inputs.after_certificate),
            Some(inputs.after_pipeline),
            Some(&format!("{} — After Remediation", inputs.title)),
        );
        std::fs::write(output_dir.join(&artifacts.before_dashboard), before_dashboard)?;
        std::fs::write(output_dir.join(&artifacts.after_dashboard), after_dashboard)?;

        let top_improvements = compute_top_improvements(
            inputs.before_scene,
            inputs.after_scene,
            inputs.before_certificate,
            inputs.after_certificate,
        );
        let lint_suggestions = collect_lint_suggestions(inputs.before_lint);
        let before = snapshot_from(inputs.before_scene, inputs.before_lint, inputs.before_certificate);
        let after = snapshot_from(inputs.after_scene, inputs.after_lint, inputs.after_certificate);
        let delta = ShowcaseDelta {
            kappa_delta: after.kappa - before.kappa,
            violation_delta: after.violations as isize - before.violations as isize,
            lint_error_delta: after.lint_errors as isize - before.lint_errors as isize,
            lint_warning_delta: after.lint_warnings as isize - before.lint_warnings as isize,
            improved_elements: top_improvements.iter().filter(|item| item.delta > 0.0).count(),
        };
        let summary = format!(
            "{} improves κ by {:.1} percentage points and reduces violations from {} to {}.",
            inputs.after_scene.name,
            delta.kappa_delta * 100.0,
            before.violations,
            after.violations,
        );

        let manifest = ShowcaseManifest {
            scenario: inputs.scenario.into(),
            title: inputs.title.into(),
            summary,
            artifacts: artifacts.clone(),
            before,
            after,
            delta,
            remediation_notes: inputs.remediation_notes.to_vec(),
            lint_suggestions,
            top_improvements,
        };

        std::fs::write(
            output_dir.join(&artifacts.bundle_manifest),
            serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".into()),
        )?;
        std::fs::write(
            output_dir.join(&artifacts.landing_page),
            self.render_landing_page(&manifest),
        )?;
        std::fs::write(
            output_dir.join(&artifacts.bundle_readme),
            self.render_bundle_readme(&manifest),
        )?;
        std::fs::write(
            output_dir.join(&artifacts.command_script),
            self.render_command_script(&manifest),
        )?;

        Ok(manifest)
    }

    fn render_landing_page(&self, manifest: &ShowcaseManifest) -> String {
        let improvements_html = manifest
            .top_improvements
            .iter()
            .take(8)
            .map(|item| {
                format!(
                    r#"<tr><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td class=\"delta positive\">+{:.1} pts</td></tr>"#,
                    escape_html(&item.name),
                    item.before_coverage * 100.0,
                    item.after_coverage * 100.0,
                    item.delta * 100.0,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let remediation_html = manifest
            .remediation_notes
            .iter()
            .map(|note| format!("<li>{}</li>", escape_html(note)))
            .collect::<Vec<_>>()
            .join("\n");

        let suggestion_html = if manifest.lint_suggestions.is_empty() {
            "<div class=\"empty\">No lint suggestions were captured for the broken scene.</div>".into()
        } else {
            manifest
                .lint_suggestions
                .iter()
                .map(|note| format!("<li>{}</li>", escape_html(note)))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let remediation_json = serde_json::to_string(&manifest.remediation_notes)
            .unwrap_or_else(|_| "[]".into());

        format!(
            r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #050b14;
      --panel: rgba(9, 18, 35, 0.92);
      --panel-soft: rgba(10, 24, 47, 0.82);
      --line: rgba(148, 163, 184, 0.18);
      --text: #e6eef9;
      --muted: #9fb3cf;
      --green: #22c55e;
      --yellow: #f59e0b;
      --red: #ef4444;
      --blue: #38bdf8;
      --violet: #8b5cf6;
      --shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background:
      radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 28%),
      radial-gradient(circle at top left, rgba(139, 92, 246, 0.15), transparent 24%),
      linear-gradient(180deg, #050b14 0%, #02060d 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{ padding: 28px; }}
    .shell {{ max-width: 1480px; margin: 0 auto; display: grid; gap: 22px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px 30px; display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.18em; color: var(--blue); font-size: 12px; font-weight: 800; margin-bottom: 10px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(32px, 4vw, 50px); line-height: 1.02; }}
    .subhead {{ color: var(--muted); font-size: 15px; line-height: 1.7; max-width: 72ch; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); color: var(--muted); font-size: 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .metric, .card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 18px; padding: 16px; }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }}
    .value {{ margin-top: 10px; font-size: 30px; font-weight: 800; }}
    .note {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .panel {{ padding: 22px; }}
    .panel h2 {{ margin: 0 0 4px; font-size: 22px; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .control-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .control-btn {{ display: inline-flex; align-items: center; gap: 8px; padding: 11px 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.045); color: var(--text); cursor: pointer; font-size: 13px; }}
    .control-btn.active {{ background: rgba(56,189,248,0.16); border-color: rgba(56,189,248,0.34); }}
    .status-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .status-chip {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: var(--muted); font-size: 12px; }}
    .status-chip strong {{ color: var(--text); }}
    .compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .snapshot {{ background: var(--panel-soft); border: 1px solid rgba(255,255,255,0.07); border-radius: 22px; padding: 18px; }}
    .snapshot h3 {{ margin: 0 0 10px; font-size: 22px; }}
    .kpi-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    .kpi {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 14px; }}
    .kpi strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 800; }}
    .full {{ background: rgba(34,197,94,0.16); color: #86efac; }}
    .partial {{ background: rgba(245,158,11,0.16); color: #fde68a; }}
    .weak {{ background: rgba(239,68,68,0.16); color: #fca5a5; }}
    .danger {{ background: rgba(239,68,68,0.16); color: #fecaca; }}
    .good {{ background: rgba(34,197,94,0.16); color: #bbf7d0; }}
    .link-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    a.button {{ display: inline-flex; align-items: center; gap: 8px; text-decoration: none; color: var(--text); padding: 10px 14px; border-radius: 12px; background: rgba(56, 189, 248, 0.14); border: 1px solid rgba(56, 189, 248, 0.3); }}
    .double {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }}
    .story-grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }}
    .story-card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 18px; padding: 18px; }}
    .story-step {{ font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--blue); }}
    .story-title {{ margin: 10px 0 8px; font-size: 24px; }}
    .story-nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    ul {{ margin: 12px 0 0; padding-left: 18px; display: grid; gap: 10px; }}
    li {{ line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; font-size: 14px; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }}
    .delta {{ font-weight: 800; }}
    .delta.positive {{ color: #86efac; }}
    .iframe-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .iframe-grid.mode-before {{ grid-template-columns: 1fr; }}
    .iframe-grid.mode-before iframe:last-child {{ display: none; }}
    .iframe-grid.mode-after {{ grid-template-columns: 1fr; }}
    .iframe-grid.mode-after iframe:first-child {{ display: none; }}
    iframe {{ width: 100%; min-height: 560px; border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; background: #02060d; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    pre {{ margin: 16px 0 0; padding: 16px; border-radius: 16px; background: rgba(2, 6, 23, 0.86); border: 1px solid rgba(255,255,255,0.06); overflow: auto; color: #cde8ff; }}
    .footer {{ text-align: center; color: var(--muted); font-size: 12px; padding-bottom: 12px; }}
    .empty {{ color: var(--muted); font-style: italic; padding-top: 10px; }}
    @media (max-width: 1100px) {{ .hero, .compare-grid, .double, .iframe-grid, .metrics {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="eyebrow">XR Affordance Verifier • Showcase Bundle</div>
        <h1>{}</h1>
        <div class="subhead">{}</div>
        <div class="pill-row">
          <span class="pill">Scenario · {}</span>
          <span class="pill">κ Δ · {:+.1} pts</span>
          <span class="pill">Violations Δ · {}</span>
          <span class="pill">Lint errors Δ · {}</span>
          <span class="pill">Improved affordances · {}</span>
        </div>
                <div class="control-row">
                    <button class="control-btn active" data-view="split">Split view</button>
                    <button class="control-btn" data-view="before">Before only</button>
                    <button class="control-btn" data-view="after">After only</button>
                    <button class="control-btn" id="jump-story">Story mode</button>
                    <button class="control-btn" id="open-before">Open broken dashboard</button>
                    <button class="control-btn" id="open-after">Open repaired dashboard</button>
                </div>
                <div class="status-strip" id="status-strip"></div>
      </div>
      <div class="metrics">
        <div class="metric"><div class="label">Before κ</div><div class="value">{:.1}%</div><div class="note">{} violations · {} lint errors</div></div>
        <div class="metric"><div class="label">After κ</div><div class="value">{:.1}%</div><div class="note">{} violations · {} lint errors</div></div>
        <div class="metric"><div class="label">Before grade</div><div class="value">{}</div><div class="note">{} findings in total</div></div>
        <div class="metric"><div class="label">After grade</div><div class="value">{}</div><div class="note">{} findings in total</div></div>
      </div>
    </section>

        <section class="panel story-grid" id="story-panel">
            <div class="story-card">
                <div class="eyebrow">Narrated remediation flow</div>
                <div id="story-step" class="story-step">Step 1</div>
                <h2 id="story-title" class="story-title">Loading remediation story…</h2>
                <p id="story-body">Use the buttons below or the keyboard to walk the audience through the broken scene, the reasoning behind the fixes, and the resulting coverage improvements.</p>
                <div class="story-nav">
                    <button class="control-btn" id="story-prev">← Previous</button>
                    <button class="control-btn" id="story-next">Next →</button>
                </div>
            </div>
            <div class="story-card">
                <div class="eyebrow">Presenter cues</div>
                <ul>
                    <li>Press <strong>1</strong>, <strong>2</strong>, or <strong>3</strong> to switch between broken, repaired, and split-screen dashboard views.</li>
                    <li>Press <strong>[</strong> or <strong>]</strong> to move backward or forward through the remediation story.</li>
                    <li>Open the broken dashboard first if you want the most dramatic narrative arc.</li>
                    <li>Finish in split view so the audience can see the before/after evidence side by side.</li>
                </ul>
            </div>
        </section>

    <section class="panel compare-grid">
      <div class="snapshot">
        <div class="eyebrow">Before remediation</div>
        <h3>{}</h3>
        <p>The intentionally broken scene surfaces accessibility regressions quickly and gives the live demo something concrete to fix.</p>
        <div class="badges">
          <span class="badge {}">{}</span>
          <span class="badge danger">{} violations</span>
        </div>
        <div class="kpi-row">
          <div class="kpi"><span class="label">Coverage κ</span><strong>{:.1}%</strong></div>
          <div class="kpi"><span class="label">Sample pass rate</span><strong>{:.1}%</strong></div>
          <div class="kpi"><span class="label">Lint errors</span><strong>{}</strong></div>
          <div class="kpi"><span class="label">Warnings</span><strong>{}</strong></div>
        </div>
        <div class="link-row">
          <a class="button" href="{}">Open dashboard</a>
          <a class="button" href="{}">Open certificate JSON</a>
          <a class="button" href="{}">Open SVG report</a>
        </div>
      </div>
      <div class="snapshot">
        <div class="eyebrow">After remediation</div>
        <h3>{}</h3>
        <p>The repaired scene keeps the same affordance identities, making the improvement diffable and easy to explain on stage.</p>
        <div class="badges">
          <span class="badge {}">{}</span>
          <span class="badge good">{} violations</span>
        </div>
        <div class="kpi-row">
          <div class="kpi"><span class="label">Coverage κ</span><strong>{:.1}%</strong></div>
          <div class="kpi"><span class="label">Sample pass rate</span><strong>{:.1}%</strong></div>
          <div class="kpi"><span class="label">Lint errors</span><strong>{}</strong></div>
          <div class="kpi"><span class="label">Warnings</span><strong>{}</strong></div>
        </div>
        <div class="link-row">
          <a class="button" href="{}">Open dashboard</a>
          <a class="button" href="{}">Open certificate JSON</a>
          <a class="button" href="{}">Open SVG report</a>
        </div>
      </div>
    </section>

    <section class="panel double">
      <div>
        <h2>What changed</h2>
        <p>Use this list as the voice-over for the live demo. Each item corresponds to a deliberate product or layout correction.</p>
        <ul>{}</ul>
      </div>
      <div>
        <h2>Broken-scene lint guidance</h2>
        <p>The original scene already contains repair hints. Use them to justify why the fixes below are not arbitrary.</p>
        <ul>{}</ul>
      </div>
    </section>

    <section class="panel">
      <h2>Top affordance improvements</h2>
      <p>These are the biggest per-element coverage gains between the broken and remediated scenes.</p>
      <table>
        <thead><tr><th>Element</th><th>Before</th><th>After</th><th>Delta</th></tr></thead>
        <tbody>{}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Live dashboard previews</h2>
      <p>Both dashboards are self-contained HTML files generated from the same scene and certificate pipeline used by the CLI.</p>
            <div id="iframe-grid" class="iframe-grid mode-split">
                <iframe id="before-frame" src="{}" title="Before remediation dashboard"></iframe>
                <iframe id="after-frame" src="{}" title="After remediation dashboard"></iframe>
      </div>
    </section>

    <section class="panel">
      <h2>Runbook</h2>
      <p>These commands are already captured in <code>{}</code>, but keeping them on the landing page makes the bundle easier to present.</p>
      <pre>{}</pre>
    </section>

    <div class="footer">Generated by XR Affordance Verifier showcase mode. All assets are local to this folder.</div>
  </div>
    <script>
        const remediationNotes = {};
        const storyState = {{ step: 0, view: 'split' }};

        function renderStatus() {{
            const host = document.getElementById('status-strip');
            host.innerHTML = [
                `<span class="status-chip"><strong>View</strong> ${{storyState.view}}</span>`,
                `<span class="status-chip"><strong>Step</strong> ${{storyState.step + 1}} / ${{Math.max(remediationNotes.length, 1)}}</span>`,
                `<span class="status-chip"><strong>Hotkeys</strong> 1 before · 2 after · 3 split · [ ] story</span>`
            ].join('');
            document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === storyState.view));
        }}

        function setView(view) {{
            storyState.view = view;
            const grid = document.getElementById('iframe-grid');
            grid.classList.remove('mode-before', 'mode-after', 'mode-split');
            grid.classList.add(`mode-${{view}}`);
            renderStatus();
        }}

        function renderStory() {{
            const note = remediationNotes[storyState.step] || 'No remediation notes available.';
            document.getElementById('story-step').textContent = `Step ${{storyState.step + 1}}`;
            document.getElementById('story-title').textContent = `Remediation move ${{storyState.step + 1}}`;
            document.getElementById('story-body').textContent = note;
            renderStatus();
        }}

        function setStoryStep(next) {{
            if (!remediationNotes.length) return;
            storyState.step = (next + remediationNotes.length) % remediationNotes.length;
            renderStory();
        }}

        document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
        document.getElementById('story-prev').addEventListener('click', () => setStoryStep(storyState.step - 1));
        document.getElementById('story-next').addEventListener('click', () => setStoryStep(storyState.step + 1));
        document.getElementById('jump-story').addEventListener('click', () => document.getElementById('story-panel').scrollIntoView({{ behavior: 'smooth', block: 'start' }}));
        document.getElementById('open-before').addEventListener('click', () => window.open('{}', '_blank'));
        document.getElementById('open-after').addEventListener('click', () => window.open('{}', '_blank'));
        document.addEventListener('keydown', event => {{
            if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
            if (event.key === '1') setView('before');
            else if (event.key === '2') setView('after');
            else if (event.key === '3') setView('split');
            else if (event.key === '[') setStoryStep(storyState.step - 1);
            else if (event.key === ']') setStoryStep(storyState.step + 1);
        }});
        renderStory();
        setView('split');
    </script>
</body>
</html>"##,
            escape_html(&manifest.title),
            escape_html(&manifest.title),
            escape_html(&manifest.summary),
            escape_html(&manifest.scenario),
            manifest.delta.kappa_delta * 100.0,
            manifest.delta.violation_delta,
            manifest.delta.lint_error_delta,
            manifest.delta.improved_elements,
            manifest.before.kappa * 100.0,
            manifest.before.violations,
            manifest.before.lint_errors,
            manifest.after.kappa * 100.0,
            manifest.after.violations,
            manifest.after.lint_errors,
            escape_html(&manifest.before.grade),
            manifest.before.total_findings,
            escape_html(&manifest.after.grade),
            manifest.after.total_findings,
            escape_html(&manifest.before.scene_name),
            grade_css_class(&manifest.before.grade),
            escape_html(&manifest.before.grade),
            manifest.before.violations,
            manifest.before.kappa * 100.0,
            ratio_percent(manifest.before.pass_count, manifest.before.sample_count),
            manifest.before.lint_errors,
            manifest.before.lint_warnings,
            escape_html(&manifest.artifacts.before_dashboard),
            escape_html(&manifest.artifacts.before_certificate),
            escape_html(&manifest.artifacts.before_svg),
            escape_html(&manifest.after.scene_name),
            grade_css_class(&manifest.after.grade),
            escape_html(&manifest.after.grade),
            manifest.after.violations,
            manifest.after.kappa * 100.0,
            ratio_percent(manifest.after.pass_count, manifest.after.sample_count),
            manifest.after.lint_errors,
            manifest.after.lint_warnings,
            escape_html(&manifest.artifacts.after_dashboard),
            escape_html(&manifest.artifacts.after_certificate),
            escape_html(&manifest.artifacts.after_svg),
            remediation_html,
            suggestion_html,
            improvements_html,
            escape_html(&manifest.artifacts.before_dashboard),
            escape_html(&manifest.artifacts.after_dashboard),
            escape_html(&manifest.artifacts.command_script),
            escape_html(&self.render_command_script(manifest)),
            remediation_json,
            escape_html(&manifest.artifacts.before_dashboard),
            escape_html(&manifest.artifacts.after_dashboard),
        )
    }

    fn render_bundle_readme(&self, manifest: &ShowcaseManifest) -> String {
        format!(
            "# {}\n\n{}\n\n## Key Files\n\n- Landing page: {}\n- Broken dashboard: {}\n- Remediated dashboard: {}\n- Bundle manifest: {}\n\n## Before vs after\n\n- Before grade: {}\n- After grade: {}\n- κ delta: {:+.1} percentage points\n- Violation delta: {}\n- Lint error delta: {}\n\n## Suggested live flow\n\n1. Open {} and frame the problem.\n2. Click through the broken dashboard ({}) to show failing affordances.\n3. Switch to the remediated dashboard ({}) and highlight the coverage lift.\n4. End with the manifest ({}) for machine-readable evidence.\n",
            manifest.title,
            manifest.summary,
            manifest.artifacts.landing_page,
            manifest.artifacts.before_dashboard,
            manifest.artifacts.after_dashboard,
            manifest.artifacts.bundle_manifest,
            manifest.before.grade,
            manifest.after.grade,
            manifest.delta.kappa_delta * 100.0,
            manifest.delta.violation_delta,
            manifest.delta.lint_error_delta,
            manifest.artifacts.landing_page,
            manifest.artifacts.before_dashboard,
            manifest.artifacts.after_dashboard,
            manifest.artifacts.bundle_manifest,
        )
    }

    fn render_command_script(&self, manifest: &ShowcaseManifest) -> String {
        format!(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n# Generated XR Affordance Verifier showcase runbook\nopen '{}'\nopen '{}'\nopen '{}'\n",
            manifest.artifacts.landing_page,
            manifest.artifacts.before_dashboard,
            manifest.artifacts.after_dashboard,
        )
    }
}

fn snapshot_from(
    scene: &SceneModel,
    lint: &LintReport,
    certificate: &CoverageCertificate,
) -> ShowcaseSnapshot {
    ShowcaseSnapshot {
        scene_name: scene.name.clone(),
        lint_errors: lint.errors().len(),
        lint_warnings: lint.warnings().len(),
        total_findings: lint.findings.len(),
        grade: format!("{:?}", certificate.grade),
        kappa: certificate.kappa,
        pass_count: certificate.samples.iter().filter(|sample| sample.is_pass()).count(),
        sample_count: certificate.samples.len(),
        verified_regions: certificate.verified_regions.len(),
        violations: certificate.violations.len(),
    }
}

fn compute_top_improvements(
    before_scene: &SceneModel,
    after_scene: &SceneModel,
    before_certificate: &CoverageCertificate,
    after_certificate: &CoverageCertificate,
) -> Vec<ElementImprovement> {
    let mut name_lookup = HashMap::new();
    for scene in [before_scene, after_scene] {
        for element in &scene.elements {
            name_lookup.insert(element.id, element.name.clone());
        }
    }

    let mut improvements = BTreeMap::<String, ElementImprovement>::new();
    for (element_id, after_cov) in &after_certificate.element_coverage {
        let before_cov = before_certificate
            .element_coverage
            .get(element_id)
            .copied()
            .unwrap_or(0.0);
        let name = name_lookup
            .get(element_id)
            .cloned()
            .unwrap_or_else(|| element_id.to_string());
        improvements.insert(
            name.clone(),
            ElementImprovement {
                name,
                before_coverage: before_cov,
                after_coverage: *after_cov,
                delta: *after_cov - before_cov,
            },
        );
    }

    let mut values = improvements.into_values().collect::<Vec<_>>();
    values.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    values
}

fn collect_lint_suggestions(report: &LintReport) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    report
        .findings
        .iter()
        .filter_map(|finding| {
            let mut text = String::new();
            if let Some(element) = &finding.element_name {
                text.push_str(element);
                text.push_str(": ");
            }
            if let Some(suggestion) = &finding.suggestion {
                text.push_str(suggestion);
            } else {
                text.push_str(&finding.message);
            }
            if seen.insert(text.clone()) {
                Some(text)
            } else {
                None
            }
        })
        .take(8)
        .collect()
}

fn ratio_percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64 * 100.0
    }
}

fn grade_css_class(grade: &str) -> &'static str {
    match grade {
        "Full" => "full",
        "Partial" => "partial",
        _ => "weak",
    }
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::certificate::{CertificateGrade, SampleVerdict};
    use xr_types::geometry::{Sphere, Volume};
    use xr_types::scene::{InteractableElement, InteractionType};

    fn sample_scene(name: &str) -> SceneModel {
        let mut scene = SceneModel::new(name);
        scene.description = "demo showcase scene".into();
        scene.add_element(
            InteractableElement::new("launch_button", [0.0, 1.2, -0.4], InteractionType::Click)
                .with_volume(Volume::Sphere(Sphere::new([0.0, 1.2, -0.4], 0.05))),
        );
        scene
    }

    fn sample_certificate(scene: &SceneModel, coverage: f64) -> CoverageCertificate {
        let element_id = scene.elements[0].id;
        CoverageCertificate {
            id: uuid::Uuid::new_v4(),
            timestamp: "2026-03-10T00:00:00Z".into(),
            protocol_version: "0.1.0".into(),
            scene_id: scene.id,
            metadata: HashMap::new(),
            grade: if coverage >= 0.99 { CertificateGrade::Full } else { CertificateGrade::Partial },
            samples: vec![SampleVerdict::pass(vec![1.7, 0.35, 0.42, 0.25, 0.18], element_id)],
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.001,
            epsilon_estimated: 0.002,
            delta: 0.05,
            kappa: coverage,
            element_coverage: [(element_id, coverage)].into_iter().collect(),
            total_time_s: 1.2,
        }
    }

    #[test]
    fn landing_page_contains_before_after_sections() {
        let before_scene = sample_scene("before");
        let after_scene = sample_scene("after");
        let before_cert = sample_certificate(&before_scene, 0.72);
        let after_cert = sample_certificate(&after_scene, 0.98);
        let lint = LintReport {
            scene_name: "demo".into(),
            elements_checked: 1,
            rules_applied: 10,
            findings: vec![],
            elapsed_ms: 1.0,
        };
        let manifest = ShowcaseManifest {
            scenario: "accessibility-remediation".into(),
            title: "XR Demo".into(),
            summary: "Summary".into(),
            artifacts: ShowcaseArtifactSet {
                bundle_manifest: "showcase.bundle.json".into(),
                landing_page: "index.html".into(),
                bundle_readme: "README.md".into(),
                command_script: "demo_commands.sh".into(),
                before_scene: "before.scene.json".into(),
                after_scene: "after.scene.json".into(),
                before_certificate: "before.certificate.json".into(),
                after_certificate: "after.certificate.json".into(),
                before_dashboard: "before.dashboard.html".into(),
                after_dashboard: "after.dashboard.html".into(),
                before_svg: "before.report.svg".into(),
                after_svg: "after.report.svg".into(),
            },
            before: snapshot_from(&before_scene, &lint, &before_cert),
            after: snapshot_from(&after_scene, &lint, &after_cert),
            delta: ShowcaseDelta {
                kappa_delta: 0.26,
                violation_delta: -1,
                lint_error_delta: -3,
                lint_warning_delta: -1,
                improved_elements: 1,
            },
            remediation_notes: vec!["Raised a control".into()],
            lint_suggestions: vec!["launch_button: move closer".into()],
            top_improvements: compute_top_improvements(&before_scene, &after_scene, &before_cert, &after_cert),
        };

        let html = ShowcaseBundleGenerator::new().render_landing_page(&manifest);
        assert!(html.contains("Before remediation"));
        assert!(html.contains("After remediation"));
        assert!(html.contains("before.dashboard.html"));
        assert!(html.contains("after.dashboard.html"));
        assert!(html.contains("XR Demo"));
    }
}
