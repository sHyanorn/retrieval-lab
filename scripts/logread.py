"""Follows a pod's log stream (server-sent events) into a file, to test the log route.

follow: GETs URL with Accept: text/event-stream and appends the data of every
event to FILE, one data line per line. It keeps the id of the last event and,
on any drop or timeout, connects again with that id in Last-Event-ID, skipping
events it already wrote (the same id and data). It stops once an event holds
--until (default the done line of scripts/logtest.py) or after --max-min
minutes. SIDE.json: start and end (UTC), bytes and lines written, every
reconnection with the id it resumed from, the first id the new connection
sent and the events it skipped, events of several data lines, ids out of
order. Lines that are not base64 or report markers are echoed to stderr, with
a progress line every minute.

Standard library only. The proxy and its CA come from the environment.

usage: python3 scripts/logread.py follow --url URL --out FILE --side SIDE.json [--last-event-id ID]
           [--until TEXT] [--max-min M] [--timeout S] [--retry-s S]
"""

import argparse
import datetime
import hashlib
import http.client
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request

DONE = "=== logtest done ==="
QUIET = re.compile(r"[A-Za-z0-9+/=]{1,76}|=== (begin64|end|report sha256) .*")
ID = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z")


def utc():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def order(eid):
    """A sort key of an event id: nanoseconds for a timestamp id, else the text itself."""
    m = ID.fullmatch(eid or "")
    if not m:
        return (1, eid or "")
    t = datetime.datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.UTC)
    return (0, int(t.timestamp()) * 10**9 + int((m[2] or "0").ljust(9, "0")))


def events(resp, last_id):
    """(id, [data lines]) of every complete event of an event stream; an event without an id field keeps the last id."""
    data, eid = [], last_id
    while True:
        raw = resp.readline()
        if not raw:
            return  # a partial event at the end is dropped
        if not raw.endswith(b"\n"):
            return
        line = raw.rstrip(b"\r\n").decode("utf-8", "replace")
        if not line:
            if data:
                yield eid, data
            data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data.append(value)
        elif field == "id" and "\0" not in value:
            eid = value


def shown(d):
    """The log line of a data line, if it is worth echoing."""
    try:
        line = json.loads(d)["line"] if d.startswith("{") else d
    except (ValueError, KeyError, TypeError):
        return d
    line = str(line).rstrip("\n")
    return None if QUIET.fullmatch(line) else line


def follow(a, err=sys.stderr):
    t0 = time.monotonic()
    deadline = t0 + a.max_min * 60
    side = {"url": a.url.split("?")[0], "start": utc(), "end": None, "stop": None, "bytes": 0, "lines": 0, "events": 0,
            "first_id": None, "last_id": None, "skipped": 0, "events_of_several_lines": 0, "repeated_ids": 0,
            "out_of_order": 0, "out_of_order_examples": [], "connections": 0, "waits": [], "reconnects": []}
    seen, last_id, top, prev = set(), a.last_event_id, None, None
    fails, next_note = 0, t0 + 60
    out = open(a.out, "ab")

    def save():
        with open(a.side, "w") as f:
            json.dump(side, f, indent=1)

    try:
        while side["stop"] is None:
            if time.monotonic() >= deadline:
                side["stop"] = "max-min"
                break
            head = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
            if last_id:
                head["Last-Event-ID"] = last_id
            conn = {"at": utc(), "resumed_from": last_id, "events_before": side["events"], "first_id": None, "skipped": 0}
            if side["events"]:
                side["reconnects"].append(conn)
            side["connections"] += 1
            reason = "eof"
            try:
                with urllib.request.urlopen(urllib.request.Request(a.url, headers=head), timeout=a.timeout) as resp:
                    for eid, data in events(resp, last_id):
                        if conn["first_id"] is None:
                            conn["first_id"] = eid
                        key = hashlib.blake2b((f"{eid}\n" + "\n".join(data)).encode(), digest_size=12).digest()
                        if key in seen:
                            side["skipped"] += 1
                            conn["skipped"] += 1
                            continue
                        seen.add(key)
                        k = order(eid)
                        if top is not None and k < top:
                            side["out_of_order"] += 1
                            if len(side["out_of_order_examples"]) < 50:
                                side["out_of_order_examples"].append([prev, eid])
                        if eid is not None and eid == prev:
                            side["repeated_ids"] += 1
                        top = k if top is None or k > top else top
                        prev = last_id = eid
                        if side["first_id"] is None:
                            side["first_id"] = eid
                        side["last_id"] = eid
                        side["events"] += 1
                        if len(data) > 1:
                            side["events_of_several_lines"] += 1
                        for d in data:
                            b = (d + "\n").encode()
                            out.write(b)
                            side["bytes"] += len(b)
                            side["lines"] += 1
                            if (s := shown(d)) is not None:
                                print(f"{utc()} | {s[:200]}", file=err, flush=True)
                        fails = 0
                        now = time.monotonic()
                        if a.until and any(a.until in d for d in data):
                            side["stop"] = "until"
                            break
                        if now >= next_note:
                            out.flush()
                            save()
                            next_note = now + 60
                            print(f"{utc()} progress: {side['events']} events, {side['bytes']} bytes, last id {last_id}, "
                                  f"{len(side['reconnects'])} reconnects", file=err, flush=True)
                        if now >= deadline:
                            side["stop"] = "max-min"
                            break
            except urllib.error.HTTPError as e:
                reason = f"http {e.code}"
            except (TimeoutError, socket.timeout):
                reason = "timeout"
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                reason = f"error {type(e).__name__}: {e}"[:200]
            conn["ended"] = side["stop"] or reason
            out.flush()
            if side["stop"] is None:
                if not side["events"]:
                    side["waits"].append({"at": utc(), "reason": reason})
                fails += 1
                print(f"{utc()} connection ended ({reason}); again from {last_id}", file=err, flush=True)
                time.sleep(min(a.retry_s * 2 ** min(fails - 1, 5), 30, max(0, deadline - time.monotonic())))
    finally:
        out.close()
        side["end"] = utc()
        side["stop"] = side["stop"] or "interrupted"
        side["waits"] = side["waits"][-50:] + ([{"dropped": len(side["waits"]) - 50}] if len(side["waits"]) > 50 else [])
        save()
    return 0 if side["stop"] == "until" else 1


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("follow")
    f.add_argument("--url", required=True)
    f.add_argument("--out", required=True, help="the file the data lines are appended to")
    f.add_argument("--side", required=True, help="the JSON record of the read")
    f.add_argument("--last-event-id", help="the id to resume after")
    f.add_argument("--until", default=DONE, help="stop once an event holds this text; empty: never")
    f.add_argument("--max-min", type=float, default=90, help="stop after this many minutes")
    f.add_argument("--timeout", type=float, default=60, help="seconds without data before connecting again")
    f.add_argument("--retry-s", type=float, default=1, help="first wait before connecting again, doubled up to 30 s")
    a = p.parse_args(argv)
    return follow(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
