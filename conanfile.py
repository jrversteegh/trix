from pathlib import Path

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Version
from conan.errors import ConanInvalidConfiguration

class TrixConan(ConanFile):
    name = "trix"
    version = "0.1.0"

    license = "MIT"
    author = "Jaap Versteegh <j.r.versteegh@gmail.com>"
    url = "https://github.com/jrversteegh/trix"
    description = "C++ Matrix/Vector library"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def validate(self):
        compiler = self.settings.compiler
        version = int(str(Version(self.settings.compiler.version)))

        if compiler == "gcc" and version < 15:
            raise ConanInvalidConfiguration("GCC needs to be verion 15 or up")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generator = "Ninja"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        src = Path(self.source_folder)
        bld = Path(self.build_folder)
        pkg = Path(self.package_folder)
        copy(self, "*.h", dst=pkg/"include", src=src/"include")
        copy(self, "*.lib", dst=pkg/"lib", src=bld)
        copy(self, "*.a", dst=pkg/"lib", src=bld)

    def package_info(self):
        self.cpp_info.libs = ["trix"]

