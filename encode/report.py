"""The small text files of step 4's output, carried out of the pod by its log.

When the job ends, whatever the outcome, scripts/pod.sh prints the files below
between marker lines, each end line with the file's SHA-256, and last the
SHA-256 of everything from the first begin line to the last end line. read()
takes the files back from a copy of the log and checks every hash. The files
on the volume stay the originals. Standard library only.

A file whose path matches a pattern of BINARY is printed in base64, in lines
of 76 characters, after a begin64 line; its end line holds the SHA-256 of its
bytes. With BINARY empty, as here, a report is the same text as before.

usage: python encode/report.py OUT              print OUT's files
       python encode/report.py --read LOG DIR   check the last report in LOG, write its files under DIR
"""

import base64
import binascii
import fnmatch
import glob
import hashlib
import os
import sys

FILES = ("summary.json", "plan.json", "clock.json", "check1/*.json", "check2/*.json", "vectors.sha256", "reference.sha256")
BINARY = ()
BEGIN, BEGIN64, END, TOTAL = "=== begin ", "=== begin64 ", "=== end ", "=== report sha256 "


def sha(b):
    return hashlib.sha256(b).hexdigest()


def paths(out):
    for pattern in FILES:
        for p in sorted(glob.glob(os.path.join(out, pattern))):
            if os.path.isfile(p):
                yield os.path.relpath(p, out)


def binary(rel):
    return any(fnmatch.fnmatch(rel, pattern) for pattern in BINARY)


def report(out):
    """The report of OUT's files; a text file without a final newline gets one in the report."""
    parts = []
    for rel in paths(out):
        with open(os.path.join(out, rel), "rb") as f:
            data = f.read()
        if binary(rel):
            parts += [f"{BEGIN64}{rel} ===\n", base64.encodebytes(data).decode("ascii"), f"{END}{rel} {sha(data)} ===\n"]
            continue
        text = data.decode("utf-8")
        parts += [f"{BEGIN}{rel} ===\n", text if not text or text.endswith("\n") else text + "\n", f"{END}{rel} {sha(data)} ===\n"]
    block = "".join(parts)
    return block + f"{TOTAL}{sha(block.encode())} ===\n"


def blocks(lines):
    """(first, last) of every report in lines: its first begin line and its total line."""
    out, start = [], 0
    for i, ln in enumerate(lines):
        if ln.startswith(TOTAL):
            begins = [j for j in range(start, i) if lines[j].startswith((BEGIN, BEGIN64))]
            out.append((begins[0] if begins else i, i))
            start = i + 1
    return out


def unpack(lines, first, last):
    """The files of the report lines[first:last + 1], {path: bytes}; ValueError if a hash differs."""
    if sha("".join(lines[first:last]).encode()) != lines[last][len(TOTAL):].split()[0]:
        raise ValueError("the report's SHA-256 differs")
    out, rel, b64, body = {}, None, False, []
    for ln in lines[first:last]:
        if rel is None:
            if not ln.startswith((BEGIN, BEGIN64)):
                raise ValueError(f"outside a file: {ln!r}")
            b64 = ln.startswith(BEGIN64)
            rel, body = ln[len(BEGIN64 if b64 else BEGIN):].rsplit(" ===", 1)[0], []
        elif ln.startswith(f"{END}{rel} "):
            want = ln.split()[-2]
            if b64:
                rows = [x.rstrip("\n") for x in body]
                if any(len(x) != 76 for x in rows[:-1]) or (rows and not 0 < len(rows[-1]) <= 76):
                    raise ValueError(f"{rel}: base64 lines are not those of the report")
                try:
                    data = base64.b64decode("".join(rows), validate=True)
                except binascii.Error as e:
                    raise ValueError(f"{rel}: not base64: {e}") from None
            else:
                data = "".join(body).encode()
                if sha(data) != want and sha(data[:-1]) == want:
                    data = data[:-1]  # the newline the report added
            if sha(data) != want:
                raise ValueError(f"{rel}: SHA-256 differs")
            out[rel], rel = data, None
        else:
            body.append(ln)
    if rel is not None:
        raise ValueError(f"{rel}: no end line")
    return out


def read(log):
    """The files of the last report in log, {path: bytes}; ValueError if a hash differs."""
    lines = log.splitlines(keepends=True)
    found = blocks(lines)
    if not found:
        raise ValueError("no report")
    return unpack(lines, *found[-1])


def read_all(log):
    """Every report in log, in order: its files, or the ValueError that its check raised."""
    lines = log.splitlines(keepends=True)
    out = []
    for first, last in blocks(lines):
        try:
            out.append(unpack(lines, first, last))
        except ValueError as e:
            out.append(e)
    return out


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
