import glob
import os
from datetime import datetime
from pathlib import Path

import tomli
from git import Repo
from invoke import task

# Run from project root directory
script_dir = Path(__file__).absolute().parent
os.chdir(script_dir)


def get_project_version_and_date():
    pyproject_toml = script_dir / "pyproject.toml"
    if not pyproject_toml.exists():
        # For some reason, during install, poetry renames pyproject.toml to pyproject.tmp...
        pyproject_toml = script_dir / "pyproject.tmp"

    # Create header with version info
    with open(pyproject_toml, "rb") as f:
        project = tomli.load(f)
        name = project["tool"]["poetry"]["name"]
        version = project["tool"]["poetry"]["version"]
    return name, version, datetime.utcnow().strftime("%Y-%m-%d")


@task
def format(ctx):
    """Run black and isort"""
    for cmd in (
        "black .",
        "isort .",
        # f"clang-format -i {' '.join(glob.glob('include/trix/*.h'))}",
        f"clang-format -i {' '.join(glob.glob('src/*.cpp'))}",
        # f"clang-format -i {' '.join(glob.glob('tests/*.cpp'))}",
        "pandoc -s -o README.md README.rst",
    ):
        ctx.run(cmd, echo=True)


@task
def check(ctx):
    """Run flake8"""
    for cmd in ("flake8 .",):
        ctx.run(cmd, echo=True)


@task
def test(ctx):
    """Run tests"""
    for cmd in (
        "pytest --cov --junitxml=build/reports/tests.xml",
        "ctest --output-junit ctest_tests.xml --test-dir build",
    ):
        ctx.run(cmd, echo=True)


@task
def update_revision(ctx):
    repo = Repo()
    ref = repo.head.object.hexsha[:8]
    date = repo.head.commit.authored_date
    with open("src/revision.txt", "w") as f:
        f.write(f"{ref} {date}\n")


@task
def create_build_dir(ctx):
    """Build"""
    for cmd in ("mkdir -p build",):
        ctx.run(cmd, echo=True)


@task(update_revision, create_build_dir)
def build(ctx):
    """Build"""
    for cmd in ("poetry build -vv",):
        ctx.run(cmd, echo=True)


@task
def clean(ctx):
    """Clean"""
    for cmd in ("rm -rf build",):
        ctx.run(cmd, echo=True)


@task(clean, build)
def rebuild(ctx): ...
