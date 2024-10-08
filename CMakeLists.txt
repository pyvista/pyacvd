cmake_minimum_required(VERSION 3.15...3.26)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  # Name of the extension
  _clustering

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # conserve space by reusing a shared libnanobind across libraries
  NB_STATIC

  src/clustering.cpp
)


# Compiler-specific options
if(MSVC)
  # Use MSVC optimization levels
  target_compile_options(_clustering PRIVATE /O2 /std:c++17)
else()
  # Assuming GCC or Clang
  target_compile_options(_clustering PRIVATE -O3)

endif()

# Example debugging
# set solib-search-path /home/user/python/.venv311/lib/python3.11/site-packages/myext/
# set breakpoint with b qual.cpp:4872
# target_compile_options(_clustering PRIVATE -g -O0)
# target_compile_options(pfh PRIVATE -g -O0)

# Install directive for scikit-build-core
install(TARGETS _clustering LIBRARY DESTINATION pyacvd)
