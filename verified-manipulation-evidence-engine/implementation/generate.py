"""
Master generation script for VMEE codebase.
Generates all module files with substantial algorithmic content.
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(BASE)

def w(path, content):
    full = os.path.join(IMPL, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    lines = len([l for l in content.splitlines() if l.strip()])
    print(f"  {path}: {lines} non-empty lines")
    return lines

total = 0

# Execute all generation parts
parts_dir = os.path.join(BASE, "_gen")
if os.path.isdir(parts_dir):
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith('.py'):
            print(f"Running {fn}...")
            exec(open(os.path.join(parts_dir, fn)).read())

print(f"\nTotal generated non-empty lines: {total}")
