from pathlib import Path


def test_scripts_use_repo_relative_paths_and_one_worker() -> None:
    scripts = [Path("scripts/setup.ps1"), Path("scripts/start.ps1"), Path("scripts/test.ps1")]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in scripts)

    assert all("$PSScriptRoot" in path.read_text(encoding="utf-8") for path in scripts)
    assert "$HOME" not in combined and "$home" not in combined
    assert "Remove-Item -Recurse" not in combined
    assert '"--workers", "1"' in scripts[1].read_text(encoding="utf-8")
