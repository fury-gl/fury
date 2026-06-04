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

### Serve the Documentation Locally

To view the generated documentation and test features like the version switcher (which require a local server to bypass CORS policies), run:

```bash
$ make serve
```

This will start an HTTP server at `http://localhost:8000` and open it in your default web browser. Press `Ctrl+C` to stop the server.
