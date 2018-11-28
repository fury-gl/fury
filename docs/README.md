# Documentation Generation

## Index

- ``source``: Contains main *.rst files
- ``tutorials``: python script describe how to use the api
- ``examples``: Fury app showcases 
- ``build``: Contains the generated documentation

## Doc generation steps:

### Installing requirements

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/optional.txt
$ pip install -U -r requirements/docs.txt
```

### Generate all the Documentation

```bash
$ make -C . clean && make -C . html
```