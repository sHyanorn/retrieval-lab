"""The small text files of step 4's output, carried out of the pod by its log.

When the job ends, whatever the outcome, scripts/pod.sh prints the files below
between marker lines, each end line with the file's SHA-256, and last the
SHA-256 of everything from the first begin line to the last end line. read()
takes the files back from a copy of the log and checks every hash. The files
on the volume stay the originals. Standard library only.

usage: python encode/report.py OUT              print OUT's files
       python encode/report.py --read LOG DIR   check the last report in LOG, write its files under DIR
"""

import glob
import hashlib
import os
import sys

FILES = ("summary.json", "plan.json", "clock.json", "check1/*.json", "check2/*.json", "vectors.sha256", "reference.sha256")
BEGIN, END, TOTAL = "=== begin ", "=== end ", "=== report sha256 "


def sha(b):
    return hashlib.sha256(b).hexdigest()


def paths(out):
    for pattern in FILES:
        for p in sorted(glob.glob(os.path.join(out, pattern))):
            if os.path.isfile(p):
                yield os.path.relpath(p, out)


def report(out):
    """The report of OUT's files; a file without a final newline gets one in the report."""
    parts = []
    for rel in paths(out):
        with open(os.path.join(out, rel), "rb") as f:
            data = f.read()
        text = data.decode("utf-8")
        parts += [f"{BEGIN}{rel} ===\n", text if not text or text.endswith("\n") else text + "\n", f"{END}{rel} {sha(data)} ===\n"]
    block = "".join(parts)
    return block + f"{TOTAL}{sha(block.encode())} ===\n"


def read(log):
    """The files of the last report in log, {path: bytes}; ValueError if a hash differs."""
    lines = log.splitlines(keepends=True)
    totals = [i for i, ln in enumerate(lines) if ln.startswith(TOTAL)]
    if not totals:
        raise ValueError("no report")
    last = totals[-1]
    begins = [i for i in range(totals[-2] + 1 if len(totals) > 1 else 0, last) if lines[i].startswith(BEGIN)]
    first = begins[0] if begins else last
    if sha("".join(lines[first:last]).encode()) != lines[last][len(TOTAL):].split()[0]:
        raise ValueError("the report's SHA-256 differs")
    files, rel, body = {}, None, []
    for ln in lines[first:last]:
        if rel is None:
            if not ln.startswith(BEGIN):
                raise ValueError(f"outside a file: {ln!r}")
            rel, body = ln[len(BEGIN):].rsplit(" ===", 1)[0], []
        elif ln.startswith(f"{END}{rel} "):
            want, data = ln.split()[-2], "".join(body).encode()
            if sha(data) != want and sha(data[:-1]) == want:
                data = data[:-1]  # the newline the report added
            if sha(data) != want:
                raise ValueError(f"{rel}: SHA-256 differs")
            files[rel], rel = data, None
        else:
            body.append(ln)
    if rel is not None:
        raise ValueError(f"{rel}: no end line")
    return files


def main(argv):
    if argv[:1] == ["--read"] and len(argv) == 3:
        with open(argv[1], encoding="utf-8") as f:
            files = read(f.read())
        for rel, data in files.items():
            path = os.path.join(argv[2], rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            print(f"{sha(data)}  {rel}")
        return 0
    if len(argv) == 1:
        sys.stdout.write(report(argv[0]))
        return 0
    print("\n".join(__doc__.strip().splitlines()[-2:]), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
