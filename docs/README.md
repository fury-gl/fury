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

#### Under Linux and OSX

```bash
$ make -C . clean && make -C . html
```

#### Under Windows

```bash
$ ./make.bat clean
$ ./make.bat html
```