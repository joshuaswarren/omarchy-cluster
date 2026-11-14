"""The omarchy-cluster Arch package: one noarch package built from a local source archive, no downloads."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGBUILD = ROOT / "packaging" / "PKGBUILD"


def var(name):
    script = f'source "{PKGBUILD}"; printf "%s\\n" "${{{name}[@]}}"'
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=True).stdout.split()


def test_one_noarch_package_named_omarchy_cluster():
    assert var("pkgname") == ["omarchy-cluster"]
    assert var("arch") == ["any"]


def test_version_matches_pyproject():
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert var("pkgver") == [version]


def test_sources_are_local_and_nothing_is_fetched():
    assert all("://" not in source for source in var("source"))
    text = PKGBUILD.read_text()
    for fetcher in ("curl", "wget", "git clone", "pip install"):
        assert fetcher not in text


def test_installs_the_wheel_and_the_licence():
    text = PKGBUILD.read_text()
    assert "python -m installer" in text
    assert "/usr/share/licenses/$pkgname/LICENSE" in text


def test_the_documented_build_extracts_the_archive():
    # `makepkg -e` skips extraction, so a fresh build directory has no source tree for build() to enter
    assert "makepkg -ef" not in PKGBUILD.read_text()
