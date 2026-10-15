# Documentation Generation

## Index

-   `docs/source/`: Maintained prose, Sphinx configuration, and templates.
-   `docs/examples/`: Maintained Python gallery inputs.
-   `docs/examples_revamped/`: Generated/prepared gallery examples; do not edit these files.
-   `docs/build/`: Generated documentation output.
-   `fury/actor/`: Runtime actor implementations.
-   `requirements/`: Dependency requirement files.
-   `docs/README.md`: This documentation build guide.

`docs/examples/_valid_examples.toml` selects the gallery inputs to prepare.

## Doc generation steps:

### Installing requirements

Run these commands from the repository root in your activated development environment.

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/optional.txt
$ pip install -U -r requirements/doc.txt
$ cd docs
```

### Generate all the Documentation

After installing the requirements and entering `docs` as shown above, run the
following commands.

#### Under Linux and macOS

```bash
$ make -C . clean && make -C . html
```

To generate the documentation without running the examples:

```bash
$ make -C . clean && make -C . html-no-examples
```
#### Under Windows

```bash
$ .\make.bat clean
$ .\make.bat html
```

To generate the documentation without running the examples:

```bash
$ .\make.bat clean
$ .\make.bat html-no-examples
```

### Serve the Documentation Locally

To view the generated documentation and test features like the version switcher (which require a local server to bypass CORS policies), run:

```bash
$ make serve
```

This will start an HTTP server at `http://localhost:8000` and open it in your default web browser. Press `Ctrl+C` to stop the server.
