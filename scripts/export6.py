"""Packs the files kept from network volume xp2tk452qe into archives that scripts/carry6.py carries out by the log.

du: the bytes (du -sb) and file counts of every part of VOL: step4, step5 and its runs, step6 and its study,
bm25, done, out, fail and tmp, data, and every entry at the top of VOL and of its parent. Names and numbers
only.

pack: the files that a checksum list names (sha256sum's format, paths relative to BASE), and no other, as one
tar archive in the order of their paths, every member with mtime 0, owner 0 and mode 0644, gzipped with time
0, cut into parts NAME.tar.gz.001, .002 and so on of at most --part-bytes; and NAME.contents.tsv, the list
itself. Every file is checked against the list as it is read; a file missing or with another SHA-256 stops it
and leaves no part. The same files give the same parts, byte for byte. Files under the list's top directories
that the list does not name are counted and left out. BASE is only read.

manifest: OUT/manifest.tsv, every file of OUT but itself with its bytes, SHA-256 and form (the contents lists
as text, the parts as base64), the manifest scripts/carry6.py print reads.

verify: reads the parts of NAME in DIR as one stream, unpacks it member by member and compares each member
with the list: missing, extra and differing members are counted and named; with --extract the members are
written under a directory. It prints counts, names, sizes and SHA-256, never a file's content.

serve: serves OUT's manifest.tsv and the files it lists, and nothing else, over HTTP at /TOKEN/<file>: GET and
HEAD, a single byte range (bytes=A-B or bytes=A-). Each request is logged by method, file, range, status and the
bytes sent, never by its path, which holds the token.

fetch: downloads from URL (the server's address and token) every file of the manifest whose name begins with
NAME., for each NAME; a transfer that is cut is resumed by a range request. Each file must have the size and
SHA-256 of its line, or it is removed and the fetch refused.

usage: python3 scripts/export6.py du --vol VOL
       python3 scripts/export6.py pack --base DIR --list FILE --name NAME --out OUT [--part-bytes N]
       python3 scripts/export6.py manifest --out OUT
       python3 scripts/export6.py verify --dir DIR --name NAME --list FILE [--extract DIR]
       python3 scripts/export6.py serve --out OUT --token TOKEN [--port 8000]
       python3 scripts/export6.py fetch --url URL --manifest FILE --dir DIR --names NAME...
"""

import argparse
import glob
import gzip
import hashlib
import http.client
import http.server
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carry6 as C  # noqa: E402  its manifest reader

BLOCK = 1 << 20
PART_BYTES = 1945 * (1 << 20)  # under GitHub's 2 GiB for one release asset
HEADER = "path\tbytes\tsha256\tform"  # scripts/carry6.py's manifest
LINE = re.compile(r"([0-9a-f]{64})  (.+)")
RANGE = re.compile(r"bytes=(\d+)-(\d*)")
AGENT = "retrieval-lab-export6"
sleep = time.sleep  # tests replace it


class Refused(Exception):
    pass


def safe(rel):
    parts = rel.split("/")
    if not rel or rel.startswith("/") or "\\" in rel or "\0" in rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"path {rel!r} is not inside the directory")
    return rel


def read_list(path):
    """[(path, sha256)] of a checksum list in sha256sum's format, sorted by path."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for ln in f.read().splitlines():
            m = LINE.fullmatch(ln)
            if not m:
                raise ValueError(f"{path}: not a checksum line: {ln[:200]!r}")
            rel = safe(m[2])
            if rel in seen:
                raise ValueError(f"{path}: {rel} listed twice")
            seen.add(rel)
            out.append((rel, m[1]))
    return sorted(out)


def du_sb(p):
    r = subprocess.run(["du", "-sb", p], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout else None


def count_files(p):
    return sum(len(names) for _, _, names in os.walk(p))


DIRS = (".", "step4", "step4/vectors", "step5", "step5/runs", "step5/study", "step5/bm25", "step6", "step6/study", "step6/bm25",
        "step6/done", "step6/out", "step6/fail", "step6/tmp", "data")


def du(vol):
    """The bytes and file counts of the parts of vol, and of every entry at the top of vol and of its parent."""
    dirs = []
    for rel in DIRS:
        p = os.path.normpath(os.path.join(vol, rel))
        ok = os.path.isdir(p)
        dirs.append({"path": rel, "exists": ok, "bytes": du_sb(p) if ok else None, "files": count_files(p) if ok else None})

    def top(d):
        return [{"name": n, "dir": os.path.isdir(os.path.join(d, n)), "bytes": du_sb(os.path.join(d, n))} for n in sorted(os.listdir(d))]

    by = {x["path"]: x["bytes"] for x in dirs}
    other = by["step6"] - by["step6/study"] if by["step6"] is not None and by["step6/study"] is not None else None
    parent = os.path.dirname(os.path.abspath(vol))
    return {"vol": vol, "dirs": dirs, "step6_other_bytes": other, "top": top(vol), "parent": parent, "parent_top": top(parent)}


class Parts:
    """A file object that writes NAME.tar.gz.001, .002 ... under out, each of at most size bytes."""

    def __init__(self, out, name, size):
        self.out, self.name, self.size = out, name, size
        self.done, self.f, self.h, self.n = [], None, None, 0

    def _open(self):
        rel = f"{self.name}.tar.gz.{len(self.done) + 1:03d}"
        self.f, self.h, self.n, self.rel = open(os.path.join(self.out, rel + ".part"), "wb"), hashlib.sha256(), 0, rel

    def _close(self):
        self.f.close()
        os.replace(os.path.join(self.out, self.rel + ".part"), os.path.join(self.out, self.rel))
        self.done.append({"path": self.rel, "bytes": self.n, "sha256": self.h.hexdigest()})
        self.f = None

    def write(self, b):
        b = memoryview(b)
        while len(b):
            if self.f is None:
                self._open()
            k = min(len(b), self.size - self.n)
            self.f.write(b[:k])
            self.h.update(b[:k])
            self.n += k
            b = b[k:]
            if self.n == self.size:
                self._close()
        return None

    def flush(self):
        pass

    def close(self):
        if self.f is not None:
            self._close()


class Hashing:
    """Reads a file and hashes what was read."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, n=-1):
        b = self.f.read(n if n is not None and n >= 0 else BLOCK)
        self.h.update(b)
        self.n += len(b)
        return b


def clean(out, name):
    for p in glob.glob(os.path.join(glob.escape(out), glob.escape(name) + ".tar.gz.*")) + [os.path.join(out, name + ".contents.tsv")]:
        if os.path.exists(p):
            os.remove(p)


def pack(base, entries, out, name, part_bytes=PART_BYTES):
    """The files of entries under base as NAME.tar.gz.NNN parts and NAME.contents.tsv in out; a summary of names,
    counts, sizes and SHA-256."""
    if part_bytes < 1:
        raise ValueError("part bytes must be at least 1")
    os.makedirs(out, exist_ok=True)
    clean(out, name)
    missing = [rel for rel, _ in entries if not os.path.isfile(os.path.join(base, rel))]
    if missing:
        raise Refused(f"{len(missing)} of {len(entries)} listed files are missing: {', '.join(missing[:5])}")
    listed = {rel for rel, _ in entries}
    tops = sorted({rel.split("/")[0] for rel in listed})
    extra = sorted(os.path.relpath(os.path.join(d, n), base) for t in tops for d, _, names in os.walk(os.path.join(base, t))
                   for n in names if os.path.relpath(os.path.join(d, n), base) not in listed)
    parts = Parts(out, name, part_bytes)
    total = 0
    try:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=parts, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as tar:
                for rel, want in entries:
                    p = os.path.join(base, rel)
                    ti = tarfile.TarInfo(rel)
                    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(p), 0, 0o644, 0, 0, "", ""
                    with open(p, "rb") as f:
                        r = Hashing(f)
                        tar.addfile(ti, r)
                    if r.h.hexdigest() != want:
                        raise Refused(f"{rel}: its SHA-256 differs from the list")
                    total += ti.size
        parts.close()
    except BaseException:
        if parts.f is not None:
            parts.f.close()
            os.remove(os.path.join(out, parts.rel + ".part"))
        clean(out, name)
        raise
    with open(os.path.join(out, name + ".contents.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{h}  {rel}\n" for rel, h in entries)
    return {"name": name, "files": len(entries), "bytes": total, "parts": parts.done,
            "extra": {"count": len(extra), "first": extra[:5]}}


def write_manifest(out):
    """OUT/manifest.tsv over every other file of OUT, sorted: contents lists as text, parts as base64."""
    lines = [HEADER]
    for rel in sorted(os.listdir(out)):
        p = os.path.join(out, rel)
        if rel == "manifest.tsv" or not os.path.isfile(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(BLOCK), b""):
                h.update(b)
        lines.append(f"{rel}\t{os.path.getsize(p)}\t{h.hexdigest()}\t{'text' if rel.endswith('.contents.tsv') else 'base64'}")
    with open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 1


class Concat:
    """The parts as one read-only stream."""

    def __init__(self, paths):
        self.paths, self.i, self.f = paths, 0, None

    def read(self, n=-1):
        out = b""
        while self.i < len(self.paths) and (n < 0 or len(out) < n):
            if self.f is None:
                self.f = open(self.paths[self.i], "rb")
            b = self.f.read(-1 if n < 0 else n - len(out))
            if not b:
                self.f.close()
                self.f, self.i = None, self.i + 1
            out += b
        return out


def verify(dir, name, entries, extract=None):
    """Compares the members of NAME's parts in dir with entries; names, counts, sizes and SHA-256 only."""
    want = dict(entries)
    paths = sorted(glob.glob(os.path.join(glob.escape(dir), glob.escape(name) + ".tar.gz.[0-9][0-9][0-9]")))
    s = {"name": name, "parts": len(paths), "listed": len(want), "members": 0, "bytes": 0, "matched": 0, "error": None}
    seen, bad, extra = set(), [], []
    nums = [int(p.rsplit(".", 1)[1]) for p in paths]
    if nums != list(range(1, len(nums) + 1)) or not nums:
        s["error"] = f"parts are not numbered 1 to {len(nums)}: {[os.path.basename(p) for p in paths][:10]}"
    else:
        try:
            with tarfile.open(fileobj=gzip.GzipFile(fileobj=Concat(paths)), mode="r|") as tar:
                for m in tar:
                    s["members"] += 1
                    rel = m.name
                    if not m.isreg():
                        extra.append(rel)
                        continue
                    f = tar.extractfile(m)
                    h, w = hashlib.sha256(), None
                    if extract is not None and rel in want:
                        p = os.path.join(extract, safe(rel))
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        w = open(p + ".part", "wb")
                    for b in iter(lambda: f.read(BLOCK), b""):
                        h.update(b)
                        s["bytes"] += len(b)
                        if w:
                            w.write(b)
                    if w:
                        w.close()
                        os.replace(p + ".part", p)
                    if rel not in want:
                        extra.append(rel)
                    elif h.hexdigest() != want[rel]:
                        bad.append(rel)
                    else:
                        s["matched"] += 1
                    seen.add(rel)
        except (OSError, EOFError, tarfile.TarError, gzip.BadGzipFile, ValueError) as e:
            s["error"] = f"{type(e).__name__}: {e}"[:300]
    missing = sorted(set(want) - seen)
    s["mismatched"] = {"count": len(bad), "first": bad[:20]}
    s["missing"] = {"count": len(missing), "first": missing[:20]}
    s["extra"] = {"count": len(extra), "first": extra[:20]}
    s["pass"] = s["error"] is None and not bad and not missing and not extra and s["matched"] == len(want)
    return s


class Handler(http.server.BaseHTTPRequestHandler):
    """GET and HEAD of the files of server(), at /TOKEN/<file>."""

    files, token = {}, ""

    def do_GET(self):
        self.send(True)

    def do_HEAD(self):
        self.send(False)

    def send(self, body):
        t, rng = time.monotonic(), self.headers.get("Range")
        prefix = f"/{self.token}/"
        name = self.path[len(prefix):] if self.path.startswith(prefix) else None
        if name not in self.files:
            self.send_error(404)
            return self.note("-", rng, 404, 0, t)
        size = os.path.getsize(self.files[name])
        a, b, code = 0, size - 1, 200
        m = RANGE.fullmatch(rng or "")
        if m and m[2] and int(m[2]) < int(m[1]):
            m = None  # not a valid range: ignored
        if m and int(m[1]) >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return self.note(name, rng, 416, 0, t)
        if m:
            a, b, code = int(m[1]), min(int(m[2]), size - 1) if m[2] else size - 1, 206
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        sent = 0
        if body:
            try:
                with open(self.files[name], "rb") as f:
                    f.seek(a)
                    while sent < b - a + 1:
                        sent += self.wfile.write(f.read(min(BLOCK, b - a + 1 - sent)))
            except OSError:  # the client went away
                pass
        self.note(name, rng, code, sent, t)

    def note(self, name, rng, code, sent, t):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {self.command} {name} {rng or '-'} {code} {sent} B "
              f"{time.monotonic() - t:.1f} s", flush=True)

    def log_message(self, *args):  # its lines hold the path, and so the token
        pass


def server(out, token, port, host="0.0.0.0", handler=Handler):
    """An HTTP server of OUT's manifest.tsv and the files it lists, each checked against its size."""
    if not re.fullmatch(r"[A-Za-z0-9]{16,}", token):
        raise ValueError("the token must be 16 or more letters and digits")
    files = {"manifest.tsv": os.path.join(out, "manifest.tsv")}
    with open(files["manifest.tsv"], encoding="utf-8") as f:
        for rel, n, _ in C.parse_manifest(f.read()):
            files[rel] = os.path.join(out, rel)
            if os.path.getsize(files[rel]) != n:
                raise Refused(f"{rel}: {os.path.getsize(files[rel])} bytes, the manifest has {n}")
    s = http.server.ThreadingHTTPServer((host, port), type("Serve", (handler,), {"files": files, "token": token}))
    s.daemon_threads = True
    return s


def fetch(url, rows, dir, tries=8, timeout=60):
    """Downloads the files of rows, [(path, bytes, sha256)], from url into dir. A cut transfer is resumed by a range
    request; after tries requests in a row without progress, or an answer 4xx, it is refused. A file whose size or
    SHA-256 differs from its row is removed and refused. A summary of sizes, times and requests."""
    os.makedirs(dir, exist_ok=True)
    done = []
    for rel, size, want in rows:
        part, t, requests, fails, why = os.path.join(dir, C.safe(rel) + ".part"), time.monotonic(), 0, 0, None
        open(part, "wb").close()
        while os.path.getsize(part) < size:
            have = before = os.path.getsize(part)
            req = urllib.request.Request(f"{url}/{rel}", headers={"User-Agent": AGENT, **({"Range": f"bytes={have}-"} if have else {})})
            requests += 1
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if r.status == 200:
                        have = 0
                    elif r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {have}-"):
                        raise OSError(f"status {r.status}, Content-Range {r.headers.get('Content-Range')}")
                    with open(part, "r+b") as f:
                        f.truncate(have)
                        f.seek(have)
                        for b in iter(lambda: r.read(BLOCK), b""):
                            f.write(b)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    os.remove(part)
                    raise Refused(f"{rel}: {e}") from None
                why = e
            except (OSError, http.client.HTTPException) as e:
                why = e
            if os.path.getsize(part) > before:
                fails = 0
                continue
            fails += 1
            if fails >= tries:
                os.remove(part)
                raise Refused(f"{rel}: no progress in {tries} requests: {type(why).__name__}: {why}"[:300])
            sleep(min(60, 5 * 2 ** (fails - 1)))
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(BLOCK), b""):
                h.update(b)
        if os.path.getsize(part) != size or h.hexdigest() != want:
            os.remove(part)
            raise Refused(f"{rel}: its SHA-256 differs from the manifest")
        os.replace(part, os.path.join(dir, rel))
        s = time.monotonic() - t
        done.append({"path": rel, "bytes": size, "seconds": round(s, 1), "mib_per_s": round(size / (1 << 20) / max(s, 1e-3), 2),
                     "requests": requests})
    return {"files": done}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("du")
    d.add_argument("--vol", required=True)
    k = sub.add_parser("pack")
    k.add_argument("--base", required=True, help="the directory the list's paths are relative to (only read)")
    k.add_argument("--list", required=True, help="checksum list, sha256sum's format")
    k.add_argument("--name", required=True)
    k.add_argument("--out", required=True, help="where the parts and the contents list are written")
    k.add_argument("--part-bytes", type=int, default=PART_BYTES)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--list", required=True)
    v.add_argument("--extract")
    s = sub.add_parser("serve")
    s.add_argument("--out", required=True)
    s.add_argument("--token", required=True)
    s.add_argument("--port", type=int, default=8000)
    f = sub.add_parser("fetch")
    f.add_argument("--url", required=True, help="the server's address and token, without a final /")
    f.add_argument("--manifest", required=True)
    f.add_argument("--dir", required=True)
    f.add_argument("--names", nargs="+", required=True)
    a = p.parse_args(argv)
    if a.cmd == "serve":
        s = server(a.out, a.token, a.port)
        print(f"serving {len(s.RequestHandlerClass.files)} files on port {a.port}", flush=True)
        s.serve_forever()
        return 1
    if a.cmd == "fetch":
        with open(a.manifest, encoding="utf-8") as f:
            rows = C.parse_manifest(f.read())
        try:
            for n in a.names:
                if not any(r.startswith(n + ".") for r, _, _ in rows):
                    raise Refused(f"no file of {n} in the manifest")
            s = fetch(a.url.rstrip("/"), [x for x in rows if any(x[0].startswith(n + ".") for n in a.names)], a.dir)
        except Refused as e:
            print(f"refused: {e}", flush=True)
            return 2
        print(f"=== export6 fetch ===\n{json.dumps(s, indent=1)}\n=== end export6 fetch ===", flush=True)
        return 0
    if a.cmd == "du":
        print("=== export6 du ===")
        print(json.dumps(du(a.vol), indent=1))
        print("=== end export6 du ===", flush=True)
        return 0
    if a.cmd == "pack":
        try:
            s = pack(a.base, read_list(a.list), a.out, a.name, a.part_bytes)
        except (Refused, ValueError) as e:
            print(f"refused: {a.name}: {e}", flush=True)
            return 2
        print(f"=== export6 pack {a.name} ===\n{json.dumps(s, indent=1)}\n=== end export6 pack {a.name} ===", flush=True)
        return 0
    if a.cmd == "manifest":
        print(f"manifest.tsv: {write_manifest(a.out)} files", flush=True)
        return 0
    s = verify(a.dir, a.name, read_list(a.list), a.extract)
    print(json.dumps(s, indent=1))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
