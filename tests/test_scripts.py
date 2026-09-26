"""Tests for scripts/check_secrets.sh: it passes on a clean repo and catches real-looking secrets."""
import random
import shutil
import string
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="git and bash required")
GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "init.defaultBranch=main"]


def make_repo(tmp_path: Path, files: dict) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_secrets.sh", repo / "scripts" / "check_secrets.sh")
    (repo / ".env.example").write_text("AI_API_KEY=\n")
    for rel, content in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    subprocess.run(GIT + ["init", "-q"], cwd=repo, check=True)
    subprocess.run(GIT + ["add", "-A"], cwd=repo, check=True)
    subprocess.run(GIT + ["commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def scan(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/check_secrets.sh"], cwd=repo, capture_output=True, text=True)


def looks_real(prefix: str, n: int) -> str:
    rng = random.Random(7)  # deterministic, random-looking, contains no allowlisted word
    return prefix + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"README.md": 'export AI_API_KEY="<PROVIDED_API_KEY>"\n',
                                "tests/test_x.py": 'KEY = "sk-test-FAKEKEY-0123456789abcdef"\n'})
    res = scan(repo)
    assert res.returncode == 0 and "SECRET SCAN: PASS" in res.stdout


@pytest.mark.parametrize("content", [
    f'GROQ = "{looks_real("gsk_", 52)}"\n',
    f'client = OpenAI(api_key="{looks_real("sk-proj-", 40)}")\n',
    f'token: "{looks_real("", 24)}"\n',
])
def test_real_looking_secret_fails(tmp_path: Path, content: str) -> None:
    res = scan(make_repo(tmp_path, {"app.py": content}))
    assert res.returncode == 1 and "SECRET PATTERN in tracked files" in res.stdout


def test_secret_only_in_history_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"app.py": f'KEY = "{looks_real("gsk_", 52)}"\n'})
    (repo / "app.py").write_text("KEY = None\n")
    subprocess.run(GIT + ["commit", "-qam", "remove key"], cwd=repo, check=True)
    res = scan(repo)
    assert res.returncode == 1 and "in git history" in res.stdout and "rotate it" in res.stdout


def test_tracked_env_file_fails(tmp_path: Path) -> None:
    res = scan(make_repo(tmp_path, {".env": "AI_API_KEY=\n"}))
    assert res.returncode == 1 and "A .env file is tracked" in res.stdout


def test_env_example_with_value_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {})
    (repo / ".env.example").write_text("AI_API_KEY=something\n")
    subprocess.run(GIT + ["commit", "-qam", "oops"], cwd=repo, check=True)
    assert "must contain exactly 'AI_API_KEY='" in scan(repo).stdout
