# pip requirements files

## Index

- [default.txt](default.txt) Default requirements
- [docs.txt](docs.txt) Documentation requirements
- [optional.txt](optional.txt) Optional requirements
- [test.txt](test.txt) Requirements for running test suite

## Examples

### Installing requirements

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
```

or

```bash
conda install --yes --file=requirements/default.txt --file=requirements/optional.txt
```

### Running the tests

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/test.txt
```

or

```bash
conda install --yes --file=requirements/default.txt --file=requirements/test.txt
```

### Running the Docs

```bash
pip install -U -r requirements/default.txt
pip install -U -r requirements/optional.txt
pip install -U -r requirements/docs.txt
```
