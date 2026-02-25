#!/usr/bin/env python3
"""
litmus-check — Single-command memory model portability checker.

Usage:
    litmus-check --target arm src/
    litmus-check --target arm myfile.c
    litmus-check --target arm --stdin < snippet.c
    echo 'code...' | litmus-check --target arm --stdin

Scans C/C++/CUDA source files for concurrency patterns and checks
whether they are portable to the target architecture.
"""

import argparse
import json
import os
import re
import sys
import glob as globmod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portcheck import ARCHITECTURES
from ast_analyzer import ASTAnalyzer

# File extensions to scan
SOURCE_EXTENSIONS = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.cu', '.cl'}

# Keywords indicating concurrency-relevant code
CONCURRENCY_KEYWORDS = re.compile(
    r'atomic|std::atomic|memory_order|__sync_|__atomic_|volatile\s|'
    r'\.store\(|\.load\(|\.exchange\(|\.fetch_|compare_exchange|'
    r'__threadfence|__syncthreads|barrier\(|fence\s|dmb\s|'
    r'smp_store_release|smp_load_acquire|READ_ONCE|WRITE_ONCE',
    re.IGNORECASE
)


def find_source_files(path):
    """Recursively find C/C++/CUDA source files."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, fnames in os.walk(path):
        # Skip hidden and build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   ('build', 'target', 'node_modules', '__pycache__')]
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in SOURCE_EXTENSIONS:
                files.append(os.path.join(root, f))
    return sorted(files)


def extract_concurrency_snippets(filepath, max_lines=60):
    """Extract concurrency-relevant code regions from a source file."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
    except (IOError, OSError):
        return []

    snippets = []
    i = 0
    while i < len(lines):
        if CONCURRENCY_KEYWORDS.search(lines[i]):
            # Extract surrounding context (up to max_lines)
            start = max(0, i - 5)
            end = min(len(lines), i + max_lines)
            snippet = ''.join(lines[start:end])
            snippets.append({
                'file': filepath,
                'start_line': start + 1,
                'end_line': end,
                'code': snippet,
            })
            i = end  # skip past this region
        else:
            i += 1
    return snippets


def check_snippet(analyzer, code, target_arch, source_arch='x86',
                  warn_unrecognized=False):
    """Check a single code snippet for portability issues."""
    results = analyzer.check_portability(code, target_arch=target_arch)

    # Get coverage info if warnings requested
    coverage_info = None
    if warn_unrecognized:
        analysis = analyzer.analyze(code)
        coverage_info = {
            'coverage_confidence': analysis.coverage_confidence,
            'warnings': [w for w in analysis.warnings
                         if 'UnrecognizedPatternWarning' in w],
            'unrecognized_ops': analysis.unrecognized_ops,
        }

    # Filter to best match only
    if results:
        seen = set()
        filtered = []
        for r in results:
            if r['pattern'] not in seen:
                seen.add(r['pattern'])
                filtered.append(r)
        return filtered, coverage_info
    return [], coverage_info


def format_result(result, filepath=None, start_line=None, verbose=False):
    """Format a single check result for terminal output."""
    pat = result['pattern']
    arch = result['target_arch']
    safe = result['safe']
    conf = result.get('confidence', 0)

    if safe:
        status = '\033[32m✓ SAFE\033[0m'
    else:
        status = '\033[31m✗ UNSAFE\033[0m'

    loc = ''
    if filepath:
        loc = f'{filepath}'
        if start_line:
            loc += f':{start_line}'
        loc += ': '

    line = f'  {loc}{status}  {pat} → {arch}'
    if conf > 0:
        line += f'  (confidence: {conf:.0%})'
    if not safe and result.get('fence_fix'):
        line += f'\n    Fix: {result["fence_fix"]}'
    return line


def main():
    parser = argparse.ArgumentParser(
        prog='litmus-check',
        description='LITMUS∞ — Check C/C++/CUDA code for memory model portability issues',
        epilog='Example: litmus-check --target arm src/',
    )
    parser.add_argument('paths', nargs='*', help='Source files or directories to scan')
    parser.add_argument('--target', '-t', required=True,
                        choices=list(ARCHITECTURES.keys()),
                        help='Target architecture to check portability against')
    parser.add_argument('--source', '-s', default='x86',
                        choices=list(ARCHITECTURES.keys()),
                        help='Source architecture (default: x86)')
    parser.add_argument('--stdin', action='store_true',
                        help='Read code from stdin instead of files')
    parser.add_argument('--json', action='store_true', dest='json_output',
                        help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--fail-on-unsafe', action='store_true', default=True,
                        help='Exit non-zero when unsafe patterns detected (default: always on)')
    parser.add_argument('--warn-unrecognized', '-w', action='store_true',
                        help='Warn when code contains concurrent operations '
                             'that do not match any known pattern')

    args = parser.parse_args()

    if args.no_color:
        # Strip ANSI codes
        os.environ['NO_COLOR'] = '1'

    analyzer = ASTAnalyzer()
    all_results = []
    coverage_warnings = []
    n_files = 0
    n_issues = 0

    if args.stdin:
        code = sys.stdin.read()
        results, cov = check_snippet(analyzer, code, args.target, args.source,
                                     args.warn_unrecognized)
        for r in results:
            all_results.append({'file': '<stdin>', 'start_line': None, **r})
            if not r['safe']:
                n_issues += 1
        if cov and cov['warnings']:
            coverage_warnings.extend(cov['warnings'])
        n_files = 1
    elif args.paths:
        files = []
        for p in args.paths:
            files.extend(find_source_files(p))

        for fpath in files:
            snippets = extract_concurrency_snippets(fpath)
            if snippets:
                n_files += 1
            for snip in snippets:
                results, cov = check_snippet(analyzer, snip['code'],
                                             args.target, args.source,
                                             args.warn_unrecognized)
                for r in results:
                    all_results.append({
                        'file': snip['file'],
                        'start_line': snip['start_line'],
                        **r,
                    })
                    if not r['safe']:
                        n_issues += 1
                if cov and cov['warnings']:
                    for w in cov['warnings']:
                        coverage_warnings.append(f"{snip['file']}:{snip['start_line']}: {w}")
    else:
        parser.error('Provide source files/directories or use --stdin')

    if args.json_output:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        if not all_results:
            print(f'No concurrency patterns detected in {n_files} file(s).')
        else:
            unsafe = [r for r in all_results if not r['safe']]
            safe = [r for r in all_results if r['safe']]

            if unsafe:
                print(f'\n\033[31m{len(unsafe)} portability issue(s) found '
                      f'({args.source} → {args.target}):\033[0m')
                for r in unsafe:
                    print(format_result(r, r.get('file'), r.get('start_line'),
                                        args.verbose))
            if safe and args.verbose:
                print(f'\n\033[32m{len(safe)} safe pattern(s):\033[0m')
                for r in safe:
                    print(format_result(r, r.get('file'), r.get('start_line'),
                                        args.verbose))

            print(f'\nSummary: {len(all_results)} pattern(s) checked, '
                  f'{n_issues} issue(s) across {n_files} file(s)')

            if coverage_warnings:
                print(f'\n\033[33m⚠ Coverage warnings ({len(coverage_warnings)}):\033[0m')
                for w in coverage_warnings:
                    print(f'  {w}')

    sys.exit(1 if n_issues > 0 else 0)


if __name__ == '__main__':
    main()
