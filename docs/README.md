# Documentation Generation

## Index

-   ``source``: Contains main *.rst files
-   ``tutorials``: python script describe how to use the api
-   ``examples``: Fury app showcases
-   ``build``: Contains the generated documentation

## Doc generation steps:

### Installing requirements

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/optional.txt
$ pip install -U -r requirements/docs.txt
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
