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

To view the generated documentation over HTTP, run from `docs`:

```bash
$ make serve
```

This will start an HTTP server at `http://localhost:8000` and open it in your default web browser. Press `Ctrl+C` to stop the server.

### Version Switcher

The switcher reads `https://fury.gl/_static/versions_switcher.json`, maintained
in `fury-website`. `FURY_DOCS_VERSION` selects an exact manifest ID, such as `dev`
or `v2.0.0`. Without it, development/post-release builds select `dev`; releases
select `v<release>`.

To build documentation identified as `dev`, run from `docs`:

```bash
$ FURY_DOCS_VERSION=dev make html-no-examples
```

Or in PowerShell:

```powershell
$env:FURY_DOCS_VERSION = "dev"
.\make.bat html-no-examples
```

Only when testing unpublished manifest changes, set `FURY_SWITCHER_JSON_URL`
to the test manifest's HTTP URL and serve it from the same origin as the docs.

### Publish a Release in the Version Switcher

In `fury-website`:

1. Publish the series directory, such as `v2.1.x/`.
2. Update its manifest entry with the latest release ID/name and series URL.
   Only the stable entry gets `"preferred": true` and the `(stable)` suffix.
3. For a new stable series, repoint `latest` and `stable` to it.

Publish the manifest before documentation that depends on it. Release builds
must unset `FURY_DOCS_VERSION` or set the exact release ID, not `dev`.
