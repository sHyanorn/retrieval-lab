import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import logread as LR  # noqa: E402
import logtest as L  # noqa: E402


def stamp(i):
    return f"2026-11-13T08:{i // 600000:02d}:{i // 10000 % 60:02d}.{i % 10000:04d}00000Z"


def encode(eid, data):
    return (f"id: {eid}\n" + "".join(f"data: {d}\n" for d in data) + "\n").encode()


class Feed:
    """A log stream as Runpod serves it: events with timestamp ids, resumed after Last-Event-ID, never ending.
    plans[n] is how connection n behaves: (events to send, then "drop" | "partial" | "stall" | "hold"),
    with "replay" k to start k events before the id asked for and "repeat" i to send event i twice."""

    def __init__(self, events, plans):
        self.events, self.plans, self.asked, self.stop = events, plans, [], threading.Event()
        feed = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                n = len(feed.asked)
                last = self.headers.get("Last-Event-ID")
                feed.asked.append((last, self.headers.get("Accept")))
                plan = feed.plans[min(n, len(feed.plans) - 1)]
                ids = [e[0] for e in feed.events]
                pos = ids.index(last) + 1 if last in ids else 0
                pos = max(0, pos - plan.get("replay", 0))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.wfile.write(b": hello\n\nretry: 1000\n\n")
                    n_send = plan.get("send", len(feed.events) - pos)
                    send = feed.events[pos:pos + n_send]
                    for k, (eid, data) in enumerate(send):
                        self.wfile.write(encode(eid, data))
                        if plan.get("repeat") == pos + k:
                            self.wfile.write(encode(eid, data))
                    self.wfile.flush()
                    if plan.get("quiet"):  # hold the connection open, silent, then send the rest
                        time.sleep(plan["quiet"])
                        for eid, data in feed.events[pos + n_send:]:
                            self.wfile.write(encode(eid, data))
                        self.wfile.flush()
                    end = plan.get("then", "hold")
                    if end == "partial":
                        self.wfile.write(b"id: x\ndata: {\"line\": \"half")
                        self.wfile.flush()
                    elif end in ("stall", "hold"):
                        while not feed.stop.wait(0.05):
                            pass
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v2/pods/test/logs?source=container"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def feed():
    made = []

    def make(events, plans):
        made.append(Feed(events, plans))
        return made[-1]

    yield make
    for f in made:
        f.close()


def args(tmp_path, url, **kw):
    a = dict(url=url, out=str(tmp_path / "log.jsonl"), side=str(tmp_path / "side.json"), last_event_id=None,
             until=LR.DONE, max_min=1, timeout=0.5, connect_timeout=5, retry_s=0.01)
    a.update(kw)
    return SimpleNamespace(**a)


def lines_events(n, done=True):
    ev = [(stamp(i), [json.dumps({"line": f"line {i}"})]) for i in range(n)]
    if done:
        ev.append((stamp(n), [json.dumps({"line": f"{LR.DONE} 123 bytes 2026-11-13T08:10:00.000Z"})]))
    return ev


def test_drops_stalls_and_repeats_leave_the_stream_whole(tmp_path, feed):
    ev = lines_events(500)
    f = feed(ev, [{"send": 120, "then": "drop"}, {"send": 100, "then": "partial", "replay": 3},
                  {"send": 150, "then": "stall", "repeat": 300}, {"replay": 1}])
    a = args(tmp_path, f.url)
    assert LR.follow(a, io.StringIO()) == 0
    got = (tmp_path / "log.jsonl").read_text().splitlines()
    assert got == [e[1][0] for e in ev]
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["stop"] == "until" and side["lines"] == len(ev) and side["events"] == len(ev)
    assert side["skipped"] == 3 + 1 + 1 and side["out_of_order"] == 0
    assert [r["resumed_from"] for r in side["reconnects"]] == [stamp(119), stamp(216), stamp(366)]
    assert [r["ended"] for r in side["reconnects"]] == ["eof", "timeout", "until"]
    assert [r["first_id"] for r in side["reconnects"]] == [stamp(117), stamp(217), stamp(366)]
    assert [x[0] for x in f.asked] == [None, stamp(119), stamp(216), stamp(366)]
    assert all(x[1] == "text/event-stream" for x in f.asked)
    assert side["bytes"] == (tmp_path / "log.jsonl").stat().st_size and side["first_id"] == stamp(0)


def test_first_request_resumes_after_the_id_given(tmp_path, feed):
    ev = lines_events(50)
    f = feed(ev, [{}])
    assert LR.follow(args(tmp_path, f.url, last_event_id=stamp(9)), io.StringIO()) == 0
    assert f.asked[0][0] == stamp(9)
    assert (tmp_path / "log.jsonl").read_text().splitlines() == [e[1][0] for e in ev[10:]]


def test_records_ids_out_of_order_and_events_of_several_lines(tmp_path, feed):
    ev = lines_events(20)
    ev[5], ev[6] = ev[6], ev[5]
    ev[8] = (ev[8][0], ['{"line": "a"}', '{"line": "b"}'])
    f = feed(ev, [{}])
    assert LR.follow(args(tmp_path, f.url), io.StringIO()) == 0
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["out_of_order"] == 1 and side["out_of_order_examples"] == [[stamp(6), stamp(5)]]
    assert side["events_of_several_lines"] == 1 and side["lines"] == 22
    assert (tmp_path / "log.jsonl").read_text().splitlines()[8:10] == ['{"line": "a"}', '{"line": "b"}']


def test_stops_after_max_min_without_the_done_line(tmp_path, feed):
    f = feed(lines_events(10, done=False), [{"then": "hold"}])
    t = time.time()
    assert LR.follow(args(tmp_path, f.url, max_min=0.03, timeout=0.3), io.StringIO()) == 1
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["stop"] == "max-min" and side["lines"] == 10 and time.time() - t < 10
    assert side["start"] < side["end"]


def test_waits_for_a_log_that_is_not_there_yet(tmp_path):
    a = args(tmp_path, "http://127.0.0.1:9/v2/pods/none/logs", max_min=0.02)
    assert LR.follow(a, io.StringIO()) == 1
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["stop"] == "max-min" and side["waits"] and not side["reconnects"] and side["lines"] == 0


def test_ids_sort_as_time():
    assert LR.order("2026-11-13T08:07:38.5Z") > LR.order("2026-11-13T08:07:38.499999999Z")
    assert LR.order("2026-11-13T08:07:39Z") > LR.order("2026-11-13T08:07:38.999999999Z")
    assert LR.order("2026-11-13T08:07:38.524572778Z")[1] % 10**9 == 524572778


def test_paced_log_read_through_drops_passes_the_check(tmp_path, feed):
    """The whole route at a small scale: paced's log as Runpod's events, a reader that loses its connection
    three times, then check-chunks on what the reader wrote."""
    out = io.StringIO()
    a = SimpleNamespace(phases="A:16:0,B:8:0", reprint=None, lead_s=0, max_print_min=45, start=None, limit_min=None, grace_min=0, scale=100)
    assert L.paced(a, out) == 0
    log = out.getvalue().splitlines()
    ev = [(stamp(i), [json.dumps({"line": ln, "stream": "stdout"})]) for i, ln in enumerate(log)]
    f = feed(ev, [{"send": 5000, "then": "drop"}, {"send": 9000, "then": "partial", "replay": 40},
                  {"send": 12000, "then": "stall", "repeat": 20000}, {"replay": 2}])
    r = args(tmp_path, f.url, max_min=2)
    assert LR.follow(r, io.StringIO()) == 0
    s = L.check_chunks([r.out], scale=100, sides=[r.side])
    assert s["pass"] and s["done"] and s["reconnects"] == 3 and s["lines"] == len(log)
    assert [(p, v["intact"], v["chunks"]) for p, v in s["phases"].items()] == [("A", 69, 69), ("B", 138, 138)]
    assert s["logs"][0]["events"] == len(log) and s["read_seconds"] > 0 and s["reprint"] is None


def test_command_line(tmp_path, feed):
    f = feed(lines_events(30), [{"send": 10, "then": "drop"}, {}])
    p = subprocess.run([sys.executable, os.path.join(HERE, "logread.py"), "follow", "--url", f.url, "--out", str(tmp_path / "o"),
                        "--side", str(tmp_path / "s.json"), "--max-min", "1", "--timeout", "1", "--retry-s", "0.01"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "line 0" in p.stderr and LR.DONE in p.stderr and "connection ended (eof)" in p.stderr
    assert len((tmp_path / "o").read_text().splitlines()) == 31


def test_a_mid_stream_silence_becomes_a_reconnect_that_resumes(tmp_path, feed):
    """A quiet gap longer than --timeout but shorter than --connect-timeout: the reader reconnects
    with Last-Event-ID, the server replays from there, and the stream is whole with no gap."""
    ev = lines_events(200)
    f = feed(ev, [{"send": 80, "quiet": 1.2, "then": "hold"}, {}])
    a = args(tmp_path, f.url, timeout=0.3, connect_timeout=5)
    err = io.StringIO()
    assert LR.follow(a, err) == 0
    assert (tmp_path / "log.jsonl").read_text().splitlines() == [e[1][0] for e in ev]
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["stop"] == "until" and side["lines"] == len(ev) and side["skipped"] == 0
    # One reconnect, resuming after the last line before the silence; the first connection timed out.
    assert [r["resumed_from"] for r in side["reconnects"]] == [stamp(79)]
    assert side["reconnects"][0]["first_id"] == stamp(80) and side["reconnects"][0]["ended"] == "until"
    assert "connection ended (timeout)" in err.getvalue() and f.asked[1][0] == stamp(79)
    # A --timeout longer than the silence: no reconnect, the gap rides through on one connection.
    f2 = feed(lines_events(200), [{"send": 80, "quiet": 0.6, "then": "hold"}])
    a2 = args(tmp_path, f2.url, timeout=2.0, connect_timeout=5, out=str(tmp_path / "l2"), side=str(tmp_path / "s2"))
    assert LR.follow(a2, io.StringIO()) == 0
    assert json.loads((tmp_path / "s2").read_text())["reconnects"] == []
