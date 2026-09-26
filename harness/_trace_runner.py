"""Standalone line tracer for the Tracer (spectrum-based fault localization).

It runs INSIDE the target repository's Python interpreter, so it is stdlib-only, Python 3.8+ compatible
(no `match`, no runtime `X | Y` types, no walrus in comprehensions) and imports nothing from `harness`.

Usage:
  python _trace_runner.py --root <repo_root> --out <json_path> [--exclude <dir>]... -- script <path.py> [args...]
  python _trace_runner.py --root <repo_root> --out <json_path> [--exclude <dir>]... -- pytest <pytest args...>

Output JSON (paths relative to root, line numbers sorted):
  {"mode": "script", "exit_code": 1, "error": null,
   "runs": [{"id": "repro", "outcome": "failed", "lines": {"toolkit/inventory.py": [12, 13, 15]}}]}
In pytest mode there is one run per test ID, and each finished test is also appended to
<json_path>.partial.jsonl, so a run that is killed on timeout still leaves usable data.

Known limits:
- child processes and multiprocessing are not traced;
- C extensions are invisible;
- threading.settrace only affects threads started after tracing begins;
- decorator lines map to no function and are dropped by the scorer.
"""
import os
import sys

# `python harness/_trace_runner.py` puts harness/ first on sys.path, where our types.py, config.py or
# testing.py would shadow the stdlib or the target repo's own modules. Remove it before anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

import argparse  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

EXIT_TRACER_ERROR = 3
ALWAYS_EXCLUDED = (".venv", "venv", "site-packages")


class Tracer(object):
    """Records (relative_path, line) pairs executed in files under root."""

    def __init__(self, root, excludes):
        self.root = os.path.realpath(root)
        self.excludes = [os.path.realpath(e if os.path.isabs(e) else os.path.join(self.root, e)) for e in excludes]
        self.cache = {}
        self.current = set()

    def rel(self, filename):
        """Repo-relative path for traced files, None for everything else (decided once per filename)."""
        if filename in self.cache:
            return self.cache[filename]
        rel = None
        if filename and not filename.startswith("<"):
            path = os.path.realpath(os.path.abspath(filename))
            if path.startswith(self.root + os.sep):
                parts = path[len(self.root) + 1:].split(os.sep)
                excluded = any(p in ALWAYS_EXCLUDED for p in parts) or any(
                    path == e or path.startswith(e + os.sep) for e in self.excludes)
                if not excluded:
                    rel = "/".join(parts)
        self.cache[filename] = rel
        return rel

    def global_trace(self, frame, event, arg):
        rel = self.rel(frame.f_code.co_filename)
        if rel is None:
            return None  # stdlib, pytest, site-packages: no line events at all (keeps overhead low)
        tracer = self

        def local_trace(frame, event, arg):
            if event == "line":
                tracer.current.add((rel, frame.f_lineno))
            return local_trace

        return local_trace

    def start(self):
        threading.settrace(self.global_trace)
        sys.settrace(self.global_trace)

    @staticmethod
    def stop():
        sys.settrace(None)
        threading.settrace(None)


def group(pairs):
    """{(path, line)} -> {path: [sorted lines]}."""
    out = {}
    for path, line in pairs:
        out.setdefault(path, []).append(line)
    return dict((path, sorted(set(lines))) for path, lines in sorted(out.items()))


def run_script(tracer, args, result):
    """Run a script as __main__ under the tracer; returns its exit code."""
    if not args:
        result["error"] = "script mode needs a path"
        return EXIT_TRACER_ERROR
    path = args[0]
    sys.argv = [path] + list(args[1:])
    sys.path.insert(0, tracer.root)
    code = 0
    tracer.start()
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        if e.code is None:
            code = 0
        elif isinstance(e.code, int):
            code = e.code
        else:
            sys.stderr.write(str(e.code) + "\n")
            code = 1
    except Exception as e:  # noqa: BLE001 - the script's own failure is data, not a tracer error
        Tracer.stop()
        traceback.print_exc()
        result["exception"] = type(e).__name__
        code = 1
    finally:
        Tracer.stop()
    result["runs"] = [{"id": "repro", "outcome": "failed" if code != 0 else "passed", "lines": group(tracer.current)}]
    return code


def make_collector(pytest, tracer, per_test, outcomes, partial_path):
    """Pytest plugin that gives every test its own line set (one pytest process for all tests)."""

    class Collector(object):
        @pytest.hookimpl(hookwrapper=True)  # hookwrapper (not wrapper=True) works on pytest 7, 8 and 9
        def pytest_runtest_protocol(self, item, nextitem):
            saved = tracer.current
            tracer.current = set()
            try:
                yield
            finally:
                per_test[item.nodeid] = tracer.current
                tracer.current = saved
                record = {"id": item.nodeid, "outcome": outcomes.get(item.nodeid, "skipped"),
                          "lines": group(per_test[item.nodeid])}
                with open(partial_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")

        def pytest_runtest_logreport(self, report):
            previous = outcomes.get(report.nodeid)
            if report.failed:
                outcomes[report.nodeid] = "failed"
            elif report.skipped and previous != "failed":
                outcomes[report.nodeid] = "skipped"
            elif previous is None:
                outcomes[report.nodeid] = "passed"

    return Collector()


def run_pytest(tracer, args, result, out_path):
    """Run pytest in-process with the Collector plugin; returns pytest's exit code."""
    try:
        import pytest
    except Exception:  # noqa: BLE001
        result["error"] = "pytest not importable"
        return EXIT_TRACER_ERROR
    per_test, outcomes = {}, {}
    partial = out_path + ".partial.jsonl"
    if os.path.exists(partial):
        os.remove(partial)
    collector = make_collector(pytest, tracer, per_test, outcomes, partial)
    sys.path.insert(0, tracer.root)
    tracer.start()
    try:
        ret = pytest.main(list(args) + ["-p", "no:cacheprovider", "-q"], plugins=[collector])
    finally:
        Tracer.stop()
    result["runs"] = [{"id": nid, "outcome": outcomes.get(nid, "skipped"), "lines": group(lines)}
                      for nid, lines in per_test.items()]
    return int(ret)


def main(argv):
    """Parse arguments, trace, and always write the JSON result."""
    if "--" not in argv:
        sys.stderr.write(__doc__)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="_trace_runner")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    opts = parser.parse_args(argv[:split])
    rest = argv[split + 1:]
    mode = rest[0] if rest else ""
    result = {"mode": mode, "exit_code": None, "error": None, "runs": []}
    code = EXIT_TRACER_ERROR
    try:
        if sys.gettrace() is not None:
            result["error"] = "another tracer is already active"
        elif mode == "script":
            code = run_script(Tracer(opts.root, opts.exclude), rest[1:], result)
        elif mode == "pytest":
            code = run_pytest(Tracer(opts.root, opts.exclude), rest[1:], result, opts.out)
        else:
            result["error"] = "unknown mode %r (expected script or pytest)" % mode
    except BaseException as e:  # noqa: BLE001 - always write the result file
        Tracer.stop()
        result["error"] = "%s: %s" % (type(e).__name__, e)
        code = EXIT_TRACER_ERROR
    finally:
        result["exit_code"] = code
        parent = os.path.dirname(os.path.abspath(opts.out))
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(opts.out, "w") as fh:
            json.dump(result, fh)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
