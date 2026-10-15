# pip requirements files

## Index

- [all.txt](all.txt) All the requirements for fury.
- [compute.txt](compute.txt) Computation requirements
- [default.txt](default.txt) Default requirements
- [dev.txt](dev.txt) Developer requirements
- [doc.txt](doc.txt) Documentation requirements
- [medical.txt](medical.txt) Medical libraries requirements
- [optional.txt](optional.txt) Optional requirements
- [plot.txt](plot.txt) Plotting requirements
- [style.txt](style.txt) Requirements for code formatting checks
- [test.txt](test.txt) Requirements for running test suite
- [typing.txt](typing.txt) Requirements for spell checks and docstring checks
- [window.txt](window.txt) Requirements for all the windowing systems

## Examples

These examples use Python 3.12. Run the commands from the repository root in the activated development environment.
See [CONTRIBUTING.rst](../CONTRIBUTING.rst) for setup instructions.
Create and activate the environment before running pip:

```bash
conda create --name fury-dev --override-channels --channel conda-forge python=3.12 pip
conda activate fury-dev
```

These generated requirements files use pip syntax; install them with pip rather
than `conda install --file`.

### Installing requirements

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
pip install -e .
```


### Running the tests

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/test.txt
pip install -e .
```


### Running the Docs

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
pip install -U -r requirements/doc.txt
pip install -e .
```
