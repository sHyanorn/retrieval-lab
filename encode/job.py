"""Step 4 of PREREGISTRATION.md, Section 11, unattended on one GPU pod.

Encodes every model and dataset in Section 12's order (plan.py), measures
the rate on the first step, cuts NQ by Section 12's rule, and runs checks 1
and 2 of Section 5 with cmd/agree. Check 1 compares SciFact's GPU vectors
with CPU vectors that scripts/reference.sh computes on the pod's CPU while
the GPU encodes, and records whether those equal step 3's checksums
(results/reference-scifact-cpu.sha256). A failed check or any error stops
the job. Everything lives under VOL; a rerun on the same VOL skips the steps
whose summary exists, and encode.py resumes a step after its last chunk; a
check failed before stops it again.

VOL/data/<dataset>                       BEIR datasets (scripts/data.sh)
VOL/dense/prebuilt/<dataset>             Pyserini's indexes (scripts/dense.sh)
VOL/step4/vectors/<dataset>/<model>      vectors, <model>.json its summary
VOL/step4/reference/<model>              CPU vectors of check 1
VOL/step4/out                            log, plan, checks, checksums, summary

Exit status: 0 done, 1 an error, 2 stopped by a check or the cut rule.

usage: python encode/job.py --vol DIR [--device cuda] [--price USD] [--deadline-h H] [--dry-run]
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import encode as E
import plan as P

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STEP3 = os.path.join(REPO, "results", "reference-scifact-cpu.sha256")
FILES = ("index", "docid", "queries/index", "queries/docid")


class Stop(Exception):
    """A check failed or the cut rule ends the session."""


def slug(model):
    return model.replace("/", "_")


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(path + ".tmp", path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Job:
    def __init__(self, a):
        self.a = a
        self.s4 = os.path.join(a.vol, "step4")
        self.out = os.path.join(self.s4, "out")
        self.data = a.data or os.path.join(a.vol, "data")
        os.makedirs(self.out, exist_ok=True)
        self.models = tuple(m for m in P.ORDER if m in (a.models or P.ORDER))
        self.datasets = tuple(a.datasets or P.TOKENS)
        self.first = self.steps(P.ORDER)[0]  # measures the rate
        self.encode_run = self._encode_run
        self.count_tokens = self._count_tokens
        self.warm_up = self._warm_up
        self.st, self.st_name = None, None
        self.ref, self.ref_log = None, None
        self.pod = os.environ.get("RUNPOD_POD_ID", "local")
        self.start = a.pod_start or time.time()
        self.key = f"{self.pod}@{int(self.start)}"
        clock = read_json(self.clock_path, {})
        self.prior = sum(v for k, v in clock.items() if k != self.key)
        self.limit_h = self.prior / 3600 + a.deadline_h
        self.plan = read_json(os.path.join(self.out, "plan.json"))
        self.clock_lock, self.ended = threading.Lock(), threading.Event()

    clock_path = property(lambda self: os.path.join(self.out, "clock.json"))

    def log(self, msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
        print(line, flush=True)
        with open(os.path.join(self.out, "job.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def elapsed_h(self):
        return (self.prior + time.time() - self.start) / 3600

    def tick(self):
        """Record this pod's hours, so that a pod after it counts them."""
        with self.clock_lock:
            clock = read_json(self.clock_path, {})
            clock[self.key] = round(time.time() - self.start, 1)
            write_json(self.clock_path, clock)

    def ticking(self):
        self.tick()
        while not self.ended.wait(self.a.tick_s):
            self.tick()

    def steps(self, nq):
        return [s for s in P.steps(nq) if s[0] in self.models and s[1] in self.datasets]

    def vec(self, m, d):
        return os.path.join(self.s4, "vectors", d, slug(m))

    def done(self, step):
        return os.path.exists(self.vec(*step) + ".json")

    def done_steps(self):
        return {s for s in self.steps(P.ORDER) if self.done(s)}

    # Inputs.

    def dataset(self, d):
        """VOL/data/d, downloaded and checked by scripts/data.sh once."""
        path, ok = os.path.join(self.data, d), os.path.join(self.data, d + ".ok")
        if not os.path.exists(ok):
            if os.path.exists(path + ".zip") and os.path.isdir(path):
                shutil.rmtree(path)  # perhaps unzipped in part
            self.sh([os.path.join(REPO, "scripts", "data.sh"), self.data, d])
            open(ok, "w").close()
        return path

    def prebuilt(self, d):
        """Pyserini's flat index of d, downloaded and checked by scripts/dense.sh once."""
        dense = os.path.join(self.a.vol, "dense")
        path = os.path.join(dense, "prebuilt", d)
        if not os.path.exists(path + ".ok"):
            shutil.rmtree(path, ignore_errors=True)
            self.sh([os.path.join(REPO, "scripts", "dense.sh"), dense, d], env={"PREBUILT_ONLY": "1"})
            open(path + ".ok", "w").close()
        return path

    def sh(self, cmd, env=None):
        self.log("run: " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO, env=dict(os.environ, **(env or {})))

    # Encoding.

    def load_model(self, name, device):
        """One model in memory at a time."""
        if self.st_name != name:
            self.st, self.st_name = None, None
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            for attempt in range(3):  # the download may fail for a moment
                try:
                    self.st, self.st_name = E.load(name, device), name
                    break
                except ValueError:
                    raise
                except Exception as e:  # noqa: BLE001 - retried, then raised
                    if attempt == 2:
                        raise
                    self.log(f"loading {name} failed ({type(e).__name__}: {e}); again in 30 s")
                    time.sleep(30)
        return self.st

    def _encode_run(self, name, data, out, **kw):
        return E.run(name, data, out, load_model=self.load_model, **kw)

    def _warm_up(self, m, data):
        """A few batches before the timed step, so that the rate leaves out CUDA's start-up."""
        st = self.load_model(m, self.a.device)
        _, texts = E.read_corpus(data, 2 * self.a.batch)
        E.encoder(st, E.MODELS[m].doc_prompt, self.a.batch)(texts)

    def _count_tokens(self, m, data, skip):
        """Tokens of the documents after the first skip, prompt and the two special tokens included, cut at the length cap."""
        st = self.load_model(m, self.a.device)
        _, texts = E.read_corpus(data)
        prompt, n = E.MODELS[m].doc_prompt, 0
        for i in range(skip, len(texts), 1024):
            enc = st.tokenizer([prompt + t for t in texts[i : i + 1024]], truncation=True, max_length=st.max_seq_length)
            n += sum(len(x) for x in enc["input_ids"])
        return n

    def encode(self, m, d):
        data, out = self.dataset(d), self.vec(m, d)
        if self.plan is None and (m, d) == self.first:
            self.warm_up(m, data)
        self.log(f"encode {m} on {d}")
        s = self.encode_run(m, data, out, device=self.a.device, batch=self.a.batch, chunk=self.a.chunk)
        s["dataset"] = d
        rel = os.path.join(d, slug(m))
        lines = "".join(f"{sha256(os.path.join(out, f))}  {rel}/{f}\n" for f in FILES)
        sums = os.path.join(self.out, "sha256", rel + ".sha256")
        os.makedirs(os.path.dirname(sums), exist_ok=True)
        with open(sums, "w", encoding="utf-8") as f:
            f.write(lines)
        write_json(out + ".json", s)
        self.tick()
        self.log(f"done {m} on {d}: {s['docs']} documents, {s['seconds']} s, resumed after {s['resumed_after']}")

    # The cut rule.

    def measure(self):
        m, d = self.first
        s = read_json(self.vec(m, d) + ".json")
        tokens = self.count_tokens(m, self.dataset(d), s["resumed_after"])
        rate = tokens / s["seconds"] * P.GFLOP[m] / P.GFLOP[P.BGE_BASE]
        el = self.elapsed_h()
        dec = P.decide(rate, el, self.a.price, self.a.line, self.limit_h, done=self.done_steps(), steps=self.steps)
        self.plan = {
            "rate": rate,
            "measured": {"model": m, "dataset": d, "tokens": tokens, "seconds": s["seconds"]},
            "elapsed_h": el,
            "price": self.a.price,
            "line": self.a.line,
            "limit_h": self.limit_h,
            "stop": dec.stop,
            "nq": list(dec.nq),
            "cuts": [dict(c, when="after the measurement") for c in dec.cuts],
            "projected_h": dec.hours,
            "projected_usd": dec.usd,
        }
        write_json(os.path.join(self.out, "plan.json"), self.plan)
        self.log(f"rate {rate:.0f} BGE-base tokens/s ({tokens} tokens of {m} on {d} in {s['seconds']} s)")
        for c in dec.cuts:
            self.log(f"cut NQ of {c['model']}: projected {c['hours']} h, {c['usd']} USD")
        nq = [m for m, d in self.steps(dec.nq) if d == "nq"]
        self.log(f"projected {dec.hours:.2f} h, {dec.usd:.2f} USD; NQ for {len(nq)} models")

    def recheck(self, m):
        """Before NQ of m: the cut rule again at the rate of the steps done, so that the pod's time limit holds."""
        done = self.done_steps()
        work = secs = 0.0
        for s in done:
            r = read_json(self.vec(*s) + ".json")
            if r["seconds"] > 0:
                work += P.base_tokens(s) * (r["docs"] - r["resumed_after"]) / r["docs"]
                secs += r["seconds"]
        rate = work / secs if secs else self.plan["rate"]
        dec = P.decide(rate, self.elapsed_h(), self.a.price, self.a.line, self.limit_h, nq=self.plan["nq"], done=done, steps=self.steps)
        if dec.stop:
            raise Stop(f"no time left for NQ of {m}: {dec.hours:.2f} h projected without NQ, limit {self.limit_h:.2f} h")
        for c in dec.cuts:
            self.log(f"cut NQ of {c['model']} before NQ of {m}: at {rate:.0f} tokens/s projected {c['hours']} h, {c['usd']} USD")
            self.plan["cuts"].append(dict(c, when=f"before NQ of {m}", rate=rate))
        if dec.cuts:
            self.plan["nq"] = list(dec.nq)
            write_json(os.path.join(self.out, "plan.json"), self.plan)
        return m in dec.nq

    # Checks.

    def agree(self, ref, vecs, all_docs):
        cmd = [self.a.agree, "-ref", ref, "-vectors", vecs] + (["-all"] if all_docs else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode not in (0, 1):
            raise RuntimeError(f"{' '.join(cmd)}: {p.stderr.strip()}")
        return dict(json.loads(p.stdout), passed=p.returncode == 0)

    def check1(self, m):
        ref, vec = os.path.join(self.s4, "reference", slug(m)), self.vec(m, "scifact")
        docs = self.agree(ref, vec, False)
        queries = self.agree(os.path.join(ref, "queries"), os.path.join(vec, "queries"), True)
        r = {"docs": docs, "queries": queries, "pass": docs["passed"] and queries["passed"]}
        write_json(os.path.join(self.out, "check1", slug(m) + ".json"), r)
        self.log(f"check 1 {m}: {'pass' if r['pass'] else 'FAIL'}, min cosine documents {docs['Min']:.7f}, queries {queries['Min']:.7f}")
        if not r["pass"]:
            raise Stop(f"check 1 failed for {m}: {docs['Below9999']} documents and {queries['Below9999']} queries below 0.9999")

    def check2(self, d):
        g = self.agree(self.prebuilt(d), self.vec(P.BGE_BASE, d), True)
        r = {"vectors": g, "pass": g["passed"]}
        write_json(os.path.join(self.out, "check2", d + ".json"), r)
        self.log(f"check 2 {d}: {'pass' if r['pass'] else 'FAIL'}, min cosine {g['Min']:.7f}")
        if not r["pass"]:
            raise Stop(f"check 2 failed on {d}: {g['Below9999']} documents below 0.9999")

    def reference_models(self):
        return [m for m in self.models if not os.path.exists(os.path.join(self.s4, "reference", slug(m) + ".json"))]

    def poll(self):
        """Start or follow the CPU reference; run the checks whose inputs are ready."""
        sums = os.path.join(self.out, "reference.sha256")
        if "scifact" in self.datasets and self.plan and self.ref is None and not os.path.exists(sums):
            threads = str(self.a.ref_threads or max(1, (os.cpu_count() or 2) // 2))
            self.ref_log = open(sums + ".tmp", "w", encoding="utf-8")
            cmd = ["nice", "-n", "10", os.path.join(REPO, "scripts", "reference.sh"), self.a.python,
                   self.dataset("scifact"), os.path.join(self.s4, "reference")] + list(self.models)
            self.log(f"CPU reference on {threads} threads: {' '.join(cmd)}")
            self.ref = subprocess.Popen(cmd, cwd=REPO, stdout=self.ref_log, env=dict(os.environ, OMP_NUM_THREADS=threads))
        if self.ref is not None and self.ref.poll() is not None:
            self.ref_log.close()
            if self.ref.returncode != 0:
                raise RuntimeError(f"scripts/reference.sh exited with {self.ref.returncode}")
            self.ref = None
            os.replace(sums + ".tmp", sums)
            r = self.reference_match()
            self.log(f"CPU reference done: {r['matches']} of {r['files']} files equal to step 3's")
        for m in self.models:
            if (os.path.exists(os.path.join(self.s4, "reference", slug(m) + ".json")) and self.done((m, "scifact"))
                    and not os.path.exists(os.path.join(self.out, "check1", slug(m) + ".json"))):
                self.check1(m)
        if P.BGE_BASE in self.models:
            for d in P.CHECK2:
                if self.done((P.BGE_BASE, d)) and not os.path.exists(os.path.join(self.out, "check2", d + ".json")):
                    self.check2(d)

    def reference_match(self):
        def read(path):
            with open(path, encoding="utf-8") as f:
                return {ln.split()[1]: ln.split()[0] for ln in f if ln.strip()}

        ours, step3 = read(os.path.join(self.out, "reference.sha256")), read(STEP3)
        return {"files": len(ours), "matches": sum(step3.get(k) == v for k, v in ours.items())}

    def wait_check1(self):
        while "scifact" in self.datasets:
            self.poll()
            if not any(not os.path.exists(os.path.join(self.out, "check1", slug(m) + ".json")) for m in self.models):
                return
            if self.ref is None and not self.reference_models():
                raise RuntimeError("check 1 cannot run: SciFact vectors missing")
            time.sleep(10)

    # The session.

    def go(self):
        self.log(f"start: pod {self.pod}, device {self.a.device}, {self.prior / 3600:.2f} h before this pod, limit {self.limit_h:.2f} h")
        for k in ("check1", "check2"):  # a check failed before stops a rerun too
            for p in sorted(glob.glob(os.path.join(self.out, k, "*.json"))):
                if not read_json(p)["pass"]:
                    raise Stop(f"check {k[-1]} failed before on this volume: {os.path.basename(p)[:-5]}")
        while True:
            if self.plan is None and self.done(self.first):
                self.measure()
            if self.plan and self.plan["stop"]:
                raise Stop(f"tier 1 alone projects to {self.plan['projected_h']:.2f} h, {self.plan['projected_usd']:.2f} USD")
            self.poll()
            todo = [s for s in self.steps(self.plan["nq"] if self.plan else P.ORDER) if not self.done(s)]
            if not todo:
                break
            m, d = todo[0]
            if d == "nq":
                self.wait_check1()
                if not self.recheck(m):
                    continue
            self.encode(m, d)
        self.wait_check1()

    def finish(self, status, reason):
        self.ended.set()
        self.tick()
        if self.ref is not None:
            self.ref.terminate()
            self.ref.wait()
        files = sorted(glob.glob(os.path.join(self.out, "sha256", "*", "*.sha256")))
        with open(os.path.join(self.out, "vectors.sha256"), "w", encoding="utf-8") as f:
            for p in files:
                with open(p, encoding="utf-8") as g:
                    f.write(g.read())
        steps = [dict(read_json(self.vec(*s) + ".json"), dataset=s[1]) for s in self.steps(P.ORDER) if self.done(s)]
        checks = lambda k: {os.path.basename(p)[:-5]: read_json(p) for p in sorted(glob.glob(os.path.join(self.out, k, "*.json")))}
        check1 = {m: c for m in self.models if (c := checks("check1").get(slug(m)))}
        ref = os.path.exists(os.path.join(self.out, "reference.sha256"))
        s = {
            "status": status,
            "reason": reason,
            "pod": self.pod,
            "device": self.a.device,
            "gpu": steps[0]["device"] if steps and "device" in steps[0] else None,
            "elapsed_h": round(self.elapsed_h(), 3),
            "plan": self.plan,
            "steps": steps,
            "not_done": [list(s) for s in self.steps(self.plan["nq"] if self.plan else P.ORDER) if not self.done(s)],
            "check1": check1,
            "check2": checks("check2"),
            "reference": self.reference_match() if ref else None,
        }
        write_json(os.path.join(self.out, "summary.json"), s)
        self.log(f"{status}{': ' + reason if reason else ''}; {len(steps)} steps done, {s['elapsed_h']} h")

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        status, reason, code = "done", "", 0
        threading.Thread(target=self.ticking, daemon=True).start()
        try:
            self.go()
        except Stop as e:
            status, reason, code = "stopped", str(e), 2
        except BaseException as e:  # noqa: BLE001 - every error ends the session with a summary
            status, reason, code = "failed", f"{type(e).__name__}: {e}", 1
            self.log(traceback.format_exc())
        self.finish(status, reason)
        return code


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vol", required=True, help="network volume directory")
    p.add_argument("--data", help="BEIR datasets (default VOL/data)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--price", type=float, default=1.19, help="USD per GPU hour")
    p.add_argument("--line", type=float, default=17.9, help="the tier's GPU line, USD")
    p.add_argument("--deadline-h", type=float, default=9.0, help="hours this pod may encode, from --pod-start")
    p.add_argument("--pod-start", type=float, help="Unix time the pod started (default now)")
    p.add_argument("--models", nargs="+", choices=P.ORDER)
    p.add_argument("--datasets", nargs="+", choices=list(P.TOKENS))
    p.add_argument("--python", default=sys.executable, help="Python for scripts/reference.sh")
    p.add_argument("--agree", default=os.path.join(REPO, "bin", "agree"))
    p.add_argument("--ref-threads", type=int, help="threads of the CPU reference (default half the CPUs)")
    p.add_argument("--tick-s", type=float, default=60, help="seconds between two records of the pod's hours")
    p.add_argument("--dry-run", action="store_true", help="MiniLM on SciFact on the CPU")
    a = p.parse_args(argv)
    if a.dry_run:
        a.device, a.models, a.datasets = "cpu", a.models or [P.MINILM], a.datasets or ["scifact"]
    return a


if __name__ == "__main__":
    sys.exit(Job(parse()).run())
