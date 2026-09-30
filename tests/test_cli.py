"""Tests for the CLI: input collection, full runs with FakeLLM, exit codes, Ctrl-C, quiet mode."""
import argparse
import io
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

import test_orchestrator as scripted
from harness import cli
from harness.llm_fake import FakeLLM
from harness.ui import RichUI

FAKE_KEY = "fake-key-for-cli-tests"


class TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("REPO", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text(f"model:\n  name: openai/test-model\noutput:\n  runs_dir: {tmp_path / 'runs'}\n  workspaces_dir: {tmp_path / 'ws'}\n"
                      "phases:\n  enable_rescue: false\n  max_fix_attempts: 1\n")
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (repo / "calc.py").write_text(scripted.CALC)
    (repo / "tests" / "test_calc.py").write_text("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
    issue = tmp_path / "issue.md"
    issue.write_text("add(1, 2) returns -1 instead of 3\n")
    return {"config": str(config), "repo": str(repo), "issue": str(issue), "tmp": tmp_path}


def use_script(monkeypatch: pytest.MonkeyPatch, script: list) -> FakeLLM:
    llm = FakeLLM(script)
    monkeypatch.setattr(cli, "make_llm", lambda cfg, ui: llm)
    return llm


def verified_script() -> list:
    return scripted.localize_and_reproduce() + [
        scripted.r(scripted.call("str_replace", path="calc.py", old_str="return a - b", new_str="return a + b")),
        scripted.r(scripted.call("finish", summary="add now adds")),
        scripted.js({"verdict": "approve", "problems": [], "confidence": "high"})]


def failing_script() -> list:
    return scripted.localize_and_reproduce(reproduced=False) + [scripted.r(scripted.call("finish", summary="nothing"))]


def base_args(env: dict, *extra: str) -> list:
    return ["--config", env["config"], "--repo", env["repo"], "--issue-file", env["issue"], "--non-interactive", *extra]


# ---------------------------------------------------------------- full runs
def test_full_run_verified(env, monkeypatch, capsys) -> None:
    use_script(monkeypatch, verified_script())
    assert cli.main(base_args(env)) == 0
    out = capsys.readouterr().out
    assert "AI Coding Harness" in out and "LOCALIZE" in out and "→ view_file calc.py" in out
    assert "VERIFIED FIX" in out and "Report:" in out and "+    return a + b" in out
    assert "Verification (attempt 1)" in out and FAKE_KEY not in out
    assert (Path(env["repo"]) / "calc.py").read_text().startswith("def add(a, b):\n    return a + b")
    reports = list((env["tmp"] / "runs").glob("*/report.md"))
    assert len(reports) == 1 and reports[0].read_text().startswith("# ✅ VERIFIED FIX")


@pytest.mark.parametrize("script, strict, expected", [
    (verified_script, True, 0), (failing_script, True, 2), (failing_script, False, 0)])
def test_exit_codes(env, monkeypatch, script, strict, expected) -> None:
    use_script(monkeypatch, script())
    assert cli.main(base_args(env, *(["--strict-exit"] if strict else []))) == expected


def test_quiet_prints_only_summary(env, monkeypatch, capsys) -> None:
    use_script(monkeypatch, verified_script())
    cli.main(base_args(env, "--quiet"))
    out = capsys.readouterr().out
    assert "VERIFIED FIX" in out and "Tokens:" in out
    assert "→ view_file" not in out and "LOCALIZE" not in out and "AI Coding Harness" not in out


def test_ctrl_c_writes_partial_report(env, monkeypatch, capsys) -> None:
    def interrupt(messages):
        raise KeyboardInterrupt

    use_script(monkeypatch, [interrupt])
    assert cli.main(base_args(env)) == 130
    out = capsys.readouterr().out
    assert "Interrupted — partial report at" in out
    assert len(list((env["tmp"] / "runs").glob("*/state.json"))) == 1


def test_interactive_asks_to_solve_another(env, monkeypatch) -> None:
    use_script(monkeypatch, verified_script())
    monkeypatch.setattr(sys, "stdin", TTY(""))
    questions = []
    monkeypatch.setattr(RichUI, "ask", lambda self, q: questions.append(q) or "n")
    args = ["--config", env["config"], "--repo", env["repo"], "--issue-file", env["issue"]]
    assert cli.main(args) == 0
    assert questions == ["Solve another issue in the same repository? [y/N]"]


# ---------------------------------------------------------------- errors
def test_missing_key(env, monkeypatch, capsys) -> None:
    monkeypatch.delenv("AI_API_KEY")
    assert cli.main(base_args(env)) == 1
    assert "No API key for openai/test-model: set AI_API_KEY or OPENAI_API_KEY" in capsys.readouterr().out


def test_self_check_needs_no_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert cli.main(["--self-check"]) == 0
    out = capsys.readouterr().out
    assert "Self-check OK" in out and "git" in out and "node" in out


def test_missing_inputs_non_interactive(env, monkeypatch, capsys) -> None:
    use_script(monkeypatch, [])
    assert cli.main(["--config", env["config"], "--non-interactive", "--issue", "x"]) == 1
    assert "No repository given" in capsys.readouterr().out
    assert cli.main(["--config", env["config"], "--repo", str(env["tmp"] / "nope"), "--issue", "x"]) == 1
    assert "Repository directory not found" in capsys.readouterr().out


def test_demo_without_fixtures(env, monkeypatch, capsys) -> None:
    use_script(monkeypatch, [])
    monkeypatch.setattr(cli, "DEMO_REPO", env["tmp"] / "missing")
    assert cli.main(["--config", env["config"], "--demo"]) == 1
    assert "Demo fixtures not found" in capsys.readouterr().out


def test_bad_config(tmp_path, capsys) -> None:
    assert cli.main(["--config", str(tmp_path / "nope.yaml")]) == 1
    assert "Config file not found" in capsys.readouterr().out


# ---------------------------------------------------------------- input collection
def ns(**kw) -> argparse.Namespace:
    base = {"repo": None, "issue": None, "issue_file": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_issue_sources(env, monkeypatch) -> None:
    ui = RichUI()
    assert cli.get_issue(ns(issue_file=env["issue"], issue="ignored"), ui, False).startswith("add(1, 2)")
    assert cli.get_issue(ns(issue="  from flag  "), ui, False) == "from flag"
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped issue text\n"))
    assert cli.get_issue(ns(), ui, False) == "piped issue text"
    with pytest.raises(cli.InputError, match="Issue file not found"):
        cli.get_issue(ns(issue_file=str(env["tmp"] / "missing.md")), ui, False)


def test_interactive_paste_until_end(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", TTY(""))
    lines = iter(["", "", "Title: bug", "details here", "END", "never read"])
    monkeypatch.setattr("builtins.input", lambda: next(lines))
    # blank lines are kept while pasting; reading stops at the END line and the result is stripped
    assert cli.get_issue(ns(), RichUI(quiet=True), True) == "Title: bug\ndetails here"


def test_interactive_paste_ctrl_d_and_reprompt(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", TTY(""))
    feeds = iter([["END"], ["real issue", EOFError]])
    current: list = []

    def fake_input():
        nonlocal current
        if not current:
            current = list(next(feeds))
        item = current.pop(0)
        if item is EOFError:
            raise EOFError
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.get_issue(ns(), RichUI(quiet=True), True) == "real issue"  # empty paste -> asked again


def test_github_issue_url(monkeypatch) -> None:
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = []

    def fake_urlopen(req, timeout):
        seen.append((req.full_url, timeout))
        return Resp(json.dumps({"title": "Crash on empty input", "body": "Steps: ..."}).encode())

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    text = cli.get_issue(ns(issue="https://github.com/octo/proj/issues/42"), RichUI(), False)
    assert text == "Crash on empty input\n\nSteps: ..."
    assert seen == [("https://api.github.com/repos/octo/proj/issues/42", 10)]

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(cli.InputError, match="Could not fetch.*offline"):
        cli.get_issue(ns(issue="https://github.com/octo/proj/issues/42"), RichUI(), False)


def test_github_issue_rate_limit_and_token(monkeypatch) -> None:
    import urllib.error
    from email.message import Message

    headers = Message()
    headers["x-ratelimit-remaining"] = "0"
    seen = []

    def limited(req, timeout):
        seen.append(req.get_header("Authorization"))
        raise urllib.error.HTTPError(req.full_url, 403, "rate limit exceeded", headers, None)

    monkeypatch.setattr(cli.urllib.request, "urlopen", limited)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(cli.InputError, match="rate limit reached.*set GITHUB_TOKEN"):
        cli.get_issue(ns(issue="https://github.com/octo/proj/issues/42"), RichUI(), False)
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    with pytest.raises(cli.InputError, match="rate limit reached\\)"):
        cli.get_issue(ns(issue="https://github.com/octo/proj/issues/42"), RichUI(), False)
    assert seen == [None, "Bearer t0ken"]


def test_git_url_detection() -> None:
    assert cli.is_git_url("https://github.com/a/b") and cli.is_git_url("git@github.com:a/b.git")
    assert cli.is_git_url("/srv/repos/project.git") and not cli.is_git_url("/home/me/project")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_clone_local_bare_repo(env) -> None:
    src = env["tmp"] / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "init.defaultBranch=main"]
    subprocess.run(git + ["init", "-q"], cwd=src, check=True)
    subprocess.run(git + ["add", "."], cwd=src, check=True)
    subprocess.run(git + ["commit", "-qm", "init"], cwd=src, check=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(env["tmp"] / "proj.git")], check=True)
    cfg = argparse.Namespace(output=argparse.Namespace(workspaces_dir=str(env["tmp"] / "ws")))
    dest = cli.resolve_repo(str(env["tmp"] / "proj.git"), cfg)
    assert dest.parent == env["tmp"] / "ws" and dest.name.startswith("proj-")
    assert (dest / "a.py").read_text() == "x = 1\n"
    with pytest.raises(cli.InputError, match="git clone failed"):
        cli.resolve_repo(str(env["tmp"] / "missing.git"), cfg)


def test_two_demos_in_the_same_second_get_separate_workspaces(tmp_path, monkeypatch) -> None:
    cfg = SimpleNamespace(output=SimpleNamespace(workspaces_dir=str(tmp_path)))
    first, _ = cli.prepare_demo(cfg)
    second, _ = cli.prepare_demo(cfg)
    assert first != second and (second / "toolkit" / "text.py").exists()
