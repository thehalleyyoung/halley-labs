#!/usr/bin/env python3
"""
For each subrepo in the halley-labs monorepo, invoke copilot-cli to:
  - Reorganise the repo layout to look like a well-planned GitHub project
  - Keep every file's actual content unchanged
  - Fix all paths in API.md, README.md, and test files
  - Run tests before + after the reorganisation and ensure nothing regresses
  - Commit and push when everything is green

Usage:
    python3 run_cleanup.py              # all repos
    python3 run_cleanup.py <repo-name>  # one repo
"""

import subprocess
import sys
import os
import re
import time

LABS = os.path.dirname(os.path.abspath(__file__))

# Read submodule paths from .gitmodules
def get_subrepos():
    gitmodules = os.path.join(LABS, ".gitmodules")
    paths = []
    with open(gitmodules) as f:
        for line in f:
            m = re.match(r'\s*path\s*=\s*(.+)', line)
            if m:
                paths.append(m.group(1).strip())
    return paths

PROMPT_TEMPLATE = """\
You are working inside the git repository at: {repo_path}

Your task has two phases.

─── PHASE 1: BASELINE ─────────────────────────────────────────────────────────
Run all tests you can find (pytest, cargo test, npm test, etc.) and record which
pass and which fail. Print a clearly labelled summary:
  BASELINE PASS: <list>
  BASELINE FAIL: <list>
If no test runner exists, note that explicitly.

─── PHASE 2: REORGANISATION ───────────────────────────────────────────────────
Reorganise the repository layout so it looks like a thoughtfully structured
open-source GitHub project, following the conventions of the language(s) used.
Rules you MUST follow:
1. Do NOT alter the content of any source file — only rename/move files and
   update references to those paths.
2. Update API.md and README.md so every path, import, and code example in them
   refers to the new locations correctly.
3. Update all test files so their imports and fixture paths reflect the new
   layout.
4. After reorganising, re-run every test from Phase 1. Every test that PASSED
   in Phase 1 must still PASS. Newly passing tests are a bonus.
5. Verify that every code example in API.md and README.md executes successfully
   (run them with the appropriate interpreter/compiler).
6. Fix any issues you find until all Phase-1-passing tests are green again.

─── PHASE 3: COMMIT & PUSH ────────────────────────────────────────────────────
Only when Phase 2 is complete and all previously-passing tests still pass:
  git add -A
  git commit -m "Reorganise repo layout for clarity and convention

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  git push

Print a final summary:
  FINAL PASS: <list>
  FINAL FAIL: <list>
  FILES MOVED: <count>
  COMMITTED: yes/no
"""

def run_copilot(repo_path, log_file, logs_dir):
    repo_name = os.path.basename(repo_path)
    prompt = PROMPT_TEMPLATE.format(repo_path=repo_path)

    cmd = [
        "copilot",
        "--yolo",
        "--autopilot",
        "--add-dir", repo_path,
        "-p", prompt,
        "--share", os.path.join(logs_dir, f"{repo_name}-session.md"),
    ]

    print(f"\n{'='*70}")
    print(f"Processing: {repo_name}")
    print(f"Log: {log_file}")
    print(f"{'='*70}")

    with open(log_file, "w") as lf:
        lf.write(f"=== {repo_name} ===\n")
        lf.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

        proc.wait()
        exit_code = proc.returncode

        lf.write(f"\nExited with code: {exit_code}\n")
        lf.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return exit_code

def main():
    subrepos = get_subrepos()

    # Allow filtering to a single repo by name
    if len(sys.argv) > 1:
        target = sys.argv[1]
        subrepos = [r for r in subrepos if r == target or os.path.basename(r) == target]
        if not subrepos:
            print(f"No repo matching '{target}' found.")
            sys.exit(1)

    logs_dir = os.path.join(LABS, "cleanup_logs")
    os.makedirs(logs_dir, exist_ok=True)

    results = {}
    for repo_name in subrepos:
        repo_path = os.path.join(LABS, repo_name)
        if not os.path.isdir(repo_path):
            print(f"SKIP (not found): {repo_path}")
            results[repo_name] = "SKIP"
            continue

        log_file = os.path.join(logs_dir, f"{repo_name}.log")
        exit_code = run_copilot(repo_path, log_file, logs_dir)
        results[repo_name] = "OK" if exit_code == 0 else f"FAIL (exit {exit_code})"

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, status in results.items():
        print(f"  {status:20s}  {name}")

    failed = [n for n, s in results.items() if not s.startswith("OK")]
    if failed:
        print(f"\n{len(failed)} repo(s) failed — see logs in {logs_dir}/")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} repo(s) completed successfully.")

if __name__ == "__main__":
    main()
