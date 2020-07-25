from os import path
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
if sys.version_info < (2, 7):
    error = """fury does not support Python {0}.{2}.
               Python 3.5 and above is required.
               Check your Python version like so:

               python3 --version

               This may be due to an out-of-date pip.
               Make sure you have pip >= 9.0.1.
               Upgrade pip like so:

               pip install --upgrade pip
               """.format(2, 7)
    sys.exit(error)

if sys.version_info < (3, 4):
    from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name='fury',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Free Unified Rendering in Python",
    long_description=readme,
    author="Eleftherios Garyfallidis",
    author_email='garyfallidis@gmail.com',
    url='https://github.com/fury-gl/fury',
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'fury': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            "fury/data/files/*",
            "fury/shaders/*"
            ]
        },
    install_requires=['numpy>=1.7.1',
                      'scipy>=0.9',
                      'vtk>=8.1.2,!=9.0.0',
                      'pillow>=5.4.1'],
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
