# Legacy setup.py shim.
#
# As of P01, project metadata (name, version, dependencies, requires-python,
# classifiers) lives in pyproject.toml's [project] table (PEP 621), which is
# the single source of truth. This file is retained as a thin entry point so
# that legacy tooling invoking `python setup.py ...` or `pip install .` without
# PEP 517 still works; calling setup() with no arguments lets setuptools read
# everything from pyproject.toml. Do NOT re-declare metadata here -- duplicate
# declarations can conflict with pyproject.toml.
import setuptools

setuptools.setup()
