### C Extension

This repository interfaces with `clustering.c` via
[nanobind](https://github.com/wjakob/nanobind) to efficiently generate C
extensions.

#### Building

Build inplace with:

```bash
pip install nanobind scikit-build-core
pip install -e . --no-build-isolation
```

Run tests with

```bash
pip install pytest
pytest -x
```

Before committing and pushing code, ensure you've run `pre-commit`. Install and run with:

```
pip install pre-commit
pre-commit install
```

Then commit. `pre-commit` will run automatically`.


#### Emacs configuration

If using emacs and helm, generate the project configuration files using `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Here's a sample configuration for C++11 on Linux:

```
pip install nanobind
export NANOBIND_INCLUDE=$(python -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")
cmake -Dnanobind_DIR=$NANOBIND_INCLUDE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="/usr/include/c++/11;/usr/include/x86_64-linux-gnu/c++/11/;/usr/lib/gcc/x86_64-linux-gnu/11/include/"
```

These will be necessary for helm and treesit to determine the locations of the header files.


#### Debug Build

This can be helpful when debugging segfaults since this extension often uses raw pointers.


Set the cmake build type to debug in `pyproject.toml``
```
[tool.scikit-build]
cmake.build-type = "Debug"
```

Set the target compile options to build debug symbols with `-g` and `-O0` in `CMakeLists.txt`:

```
target_compile_options(_utilities PRIVATE -g -O0)
target_compile_options(pfh PRIVATE -g -O0)
```

Finally, run using `gdb`. For example:

```
$ gdb --args python test_ext.py
(gdb) b qual.cpp:4872
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (qual.cpp:4872) pending.
(gdb) run
Thread 1 "python" hit Breakpoint 1, ComputeWeights<float> (offset=0x1fe5830, neigh=0x20432e0,
indices=0x1bb8ec0, points=0x1d6a7a0, n_neigh=108, n_points=27, fac=-0.75, num_threads=4) at /home/user/library-path/src/qual.cpp:4872
4872      T *weights = new T[n_neigh];
(gdb)
```

#### Debugging memory leaks

These can be challenging to find. Use [valgrind](https://valgrind.org/) with the following to identify memory leaks. Be sure to download [valgrind-python.supp](https://github.com/python/cpython/blob/main/Misc/valgrind-python.supp).

```
 valgrind --leak-check=full --log-file=val.txt --suppressions=valgrind-python.supp pytest -k clus && grep 'new\[\]' val.txt
 ```

