"""The files of step 6's output as self-checking reports in a log.

encode/report.py's format and checks, over VOL/step6/out. Each call prints
one report of the files it is given: a report checks itself, so a route that
carries the files out of the pod can print any subset of them, in pieces of
any size, and read back every report that arrived whole.

Three groups: the small label-free files (the default, which the step-6 pod
prints at exit), the label-free checksum lists, and the files that hold a
label-based measure: collect's per-query files and grid.tsv, gzipped by the
job, and step 5's per-query files. Every file outside the two label-free
groups is printed in base64 only, so that no measure stands as text in a
log; the reader writes files and prints nothing of their content. The job
writes manifest.tsv, every file of the three groups but itself with its
size, SHA-256 and form, so that another pod can print them from the
manifest and the files alone.

usage: python3 step6/report.py OUT [--sums | --measures]   one report of a group
       python3 step6/report.py OUT --files REL...           one report of the files given, relative to OUT
       python3 step6/report.py OUT --list                   every file of the groups: path, bytes, text or base64
       python3 step6/report.py --read LOG DIR               every report in LOG that checks, its files under DIR
"""

import argparse
import glob
import importlib.util
import os
import sys

# encode/report.py under a name of its own, so that this module, also named report, can import it.
_spec = importlib.util.spec_from_file_location(
    "encode_report", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encode", "report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

TEXT = ("summary.json", "manifest.tsv", "environment.json", "units.tsv", "results/select.tsv", "neighbour.tsv", "dense-times.jsonl")
SUMS = ("runs.sha256", "perquery.sha256", "results/per-query.sha256")
MEASURES = ("grid.tsv.gz", "results/*/*.tsv.gz", "step5/study/*/*/exact.tsv", "step5/bm25/*.tsv")
LABEL_FREE = TEXT + SUMS


def group(out, patterns):
    """The files of out that match patterns, relative to out, in the order of the patterns."""
    rels = []
    for pattern in patterns:
        rels += [os.path.relpath(p, out) for p in sorted(glob.glob(os.path.join(out, pattern))) if os.path.isfile(p)]
    return rels


def manifest(out):
    """The lines of manifest.tsv: every file of the three groups but the manifest, its bytes, SHA-256 and form."""
    lines = ["path\tbytes\tsha256\tform"]
    for rel in group(out, LABEL_FREE + MEASURES):
        if rel != "manifest.tsv":
            with open(os.path.join(out, rel), "rb") as f:
                data = f.read()
            lines.append(f"{rel}\t{len(data)}\t{R.sha(data)}\t{'text' if rel in LABEL_FREE else 'base64'}")
    return "\n".join(lines) + "\n"


def report(out, rels):
    """One report of the files rels of out; every file not label-free in base64. A missing file is an error."""
    missing = [r for r in rels if not os.path.isfile(os.path.join(out, r))]
    if missing:
        raise FileNotFoundError(f"not in {out}: {', '.join(missing)}")
    R.FILES = tuple(glob.escape(r) for r in rels)
    R.BINARY = tuple(glob.escape(r) for r in rels if r not in LABEL_FREE)
    return R.report(out)


def read_all(log):
    """Every report of log, in order: its files, or a ValueError for a report that does not check or for the
    lines of an incomplete report before one that does. Each report's start is the begin line, searched back
    from its total line, from which the report checks; so a report cut short does not spoil the next."""
    lines = log.splitlines(keepends=True)
    out, start = [], 0
    for i, ln in enumerate(lines):
        if not ln.startswith(R.TOTAL):
            continue
        begins = [j for j in range(start, i) if lines[j].startswith((R.BEGIN, R.BEGIN64))]
        found = None
        for j in reversed(begins or [i]):
            try:
                found = (j, R.unpack(lines, j, i))
                break
            except ValueError:
                continue
        if found is None:
            out.append(ValueError(f"the report that ends at line {i + 1} does not check"))
        else:
            if begins and begins[0] < found[0]:
                out.append(ValueError(f"lines {begins[0] + 1} to {found[0]} hold an incomplete report"))
            out.append(found[1])
        start = i + 1
    return out


def read(log, dir):
    """Writes the files of every report in log that checks under dir; returns the reports' outcomes."""
    found = read_all(log)
    outcomes = []
    for i, files in enumerate(found, 1):
        if isinstance(files, ValueError):
            outcomes.append((i, str(files), []))
            continue
        for rel, data in files.items():
            path = os.path.join(dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        outcomes.append((i, None, sorted((R.sha(d), rel) for rel, d in files.items())))
    return outcomes


def main(argv):
    if argv[:1] == ["--read"]:
        if len(argv) != 3:
            print(__doc__.strip().splitlines()[-1], file=sys.stderr)
            return 2
        with open(argv[1], encoding="utf-8") as f:
            outcomes = read(f.read(), argv[2])
        for i, err, files in outcomes:
            print(f"report {i}: " + (f"does not check: {err}" if err else f"{len(files)} files"))
            for h, rel in files:
                print(f"{h}  {rel}")
        return 0 if outcomes and not any(err for _, err, _ in outcomes) else 1
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sums", action="store_true", help="the label-free checksum lists")
    g.add_argument("--measures", action="store_true", help="the files that hold a measure, in base64")
    g.add_argument("--files", nargs="+", metavar="REL", help="these files, relative to OUT")
    g.add_argument("--list", action="store_true", help="every file of the three groups, without content")
    a = p.parse_args(argv)
    if a.list:
        for rel in group(a.out, LABEL_FREE + MEASURES):
            print(f"{rel}\t{os.path.getsize(os.path.join(a.out, rel))}\t{'text' if rel in LABEL_FREE else 'base64'}")
        return 0
    rels = a.files or group(a.out, SUMS if a.sums else MEASURES if a.measures else TEXT)
    sys.stdout.write(report(a.out, rels))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
