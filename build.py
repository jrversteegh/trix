import configparser
import contextlib
import errno
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import tomli
from setuptools import Command, Extension
from setuptools.command import build_ext

module_name = "trixx"

if "CXX" in os.environ:
    compiler = os.environ["CXX"]
else:
    compiler = "g++-15"

on_windows = platform.system().startswith("Win")
script_dir = Path(__file__).absolute().parent
source_dir = script_dir / "src" / "trixx"
trixx_dir = source_dir
build_dir = script_dir / "build"
python = sys.executable
python_dir = Path(os.path.dirname(python))
cmake = python_dir / "cmake"


def get_project_version_and_date():
    pyproject_toml = script_dir / "pyproject.toml"
    if not pyproject_toml.exists():
        # For some reason, during install, poetry renames pyproject.toml to pyproject.tmp...
        pyproject_toml = script_dir / "pyproject.tmp"

    # Create header with version info
    with open(pyproject_toml, "rb") as f:
        project = tomli.load(f)
        version = project["tool"]["poetry"]["version"]
    return version, datetime.utcnow().strftime("%Y-%m-%d")


@contextlib.contextmanager
def dir_context(new_dir):
    previous_dir = os.getcwd()
    try:
        os.makedirs(new_dir)
    except FileExistsError:
        ...
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def build_module(build_type, config=""):
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    win_flags = "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW" if on_windows else ""
    config_flag = f"--config {build_type}" if on_windows else ""
    version, date = get_project_version_and_date()
    with dir_context(build_dir):
        if os.system(
            f"conan install -of conan --profile={script_dir}/conan/trix.profile --build=missing -s build_type={build_type} {script_dir}/conanfile.txt"
        ):
            raise Exception("Failed to run conan")
        if os.system(
            f"{cmake} -DCMAKE_BUILD_TYPE={build_type}"
            f" -DVERSION={version} -DDATE={date} -DBUILD_PYTHON=1 -DBUILD_TESTS=1 -DBUILD_SHARED=1"
            f" -DCMAKE_CXX_COMPILER={compiler} -G Ninja"
            f" -DCMAKE_TOOLCHAIN_FILE=conan/conan_toolchain.cmake"
            f"{config} {win_flags} {script_dir}"
        ):
            raise Exception("Failed to configure with cmake")
        if os.system(f"{cmake} --build . {config_flag} --verbose --parallel 4"):
            raise Exception("Failed to build C++ module")


class CopyCommand(Command):
    user_options = [("inplace", None, "inplace")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_modules = args[0].ext_modules

    def initialize_options(self):
        self.editable_mode = False
        self.build_lib = None
        self.inplace = None

    def finalize_options(self):
        self.set_undefined_options("build_py", ("build_lib", "build_lib"))

    def get_outputs(self):
        result = []
        for ext in self.ext_modules:
            for module_file in ext.extra_objects:
                result.append(module_file)
        return result

    def get_source_files(self):
        return [str(f) for f in trixx_dir.glob("*.*")]

    def run(self):
        for module_file in self.get_outputs():
            self.copy_file(module_file, source_dir)
            self.copy_file(module_file, self.build_lib)


def build(setup_kwargs):
    build_type = (
        "Debug"
        if os.path.exists(script_dir / ".debug") or "DEBUG" in os.environ
        else "Release"
    )
    output_dir = build_dir / build_type if on_windows else build_dir
    build_module(build_type)
    ext_modules = [
        Extension(
            name="trixx",
            sources=[],
            extra_objects=list(output_dir.glob(f"{module_name}.cpython*.*")),
        ),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": CopyCommand},
            "zip_safe": False,
        }
    )
