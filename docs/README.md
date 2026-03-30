# Documentation Generation

## Index

-   ``source``: Contains main *.rst files
-   ``tutorials``: Python scripts describing how to use the api
-   ``examples``: FURY application showcases
-   ``build``: Contains the generated documentation

## Documentation generation steps:
----------------------------------

### Installing requirements

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/optional.txt
$ pip install -U -r requirements/doc.txt
```

### Generate all the Documentation

Go to the `docs` folder and run the following commands to generate it.

#### Under Linux and OSX

```bash
$ make -C . clean && make -C . html
```

To generate the documentation without running the examples:

```bash
$ make -C . clean && make -C . html-no-examples
```
#### Under Windows

```bash
$ make clean
$ make html
```

To generate the documentation without running the examples:

```bash
$ make clean
$ make html-no-examples
```
