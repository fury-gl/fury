#!/usr/bin/env python
"""Script to auto-generate our API docs.
"""
# stdlib imports
import sys
from os.path import join as pjoin

# local imports
from apigen import ApiDocWriter

# version comparison
from packaging.version import parse

# *****************************************************************************


def abort(error):
    print('*WARNING* API documentation not generated: %s' % error)
    exit()


def generate_api_reference_rst(
    app=None, package='fury', outdir='reference', defines=True
):
    try:
        __import__(package)
    except ImportError:
        abort('Can not import ' + package)

    module = sys.modules[package]
    installed_version = parse(module.__version__)
    print('Generation API for {} v{}'.format(package, installed_version))

    docwriter = ApiDocWriter(package, rst_extension='.rst', other_defines=defines)
    docwriter.package_skip_patterns += [
        r'.*test.*$',
        # r'^\.utils.*',
        r'\._version.*$',
        r'\.interactor.*$',
        r'\.optpkg.*$',
    ]
    docwriter.object_skip_patterns += [
        r'.*FetcherError.*$',
        r'.*urlopen.*',
        r'.*add_callback.*',
    ]
    if app is not None:
        outdir = pjoin(app.builder.srcdir, outdir)

    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'index', relative_to=outdir)
    print('%d files written' % len(docwriter.written_modules))


def setup(app):
    """Setup sphinx extension for API reference generation."""
    app.connect('builder-inited', generate_api_reference_rst)
    # app.connect('build-finished', summarize_failing_examples)

    metadata = {'parallel_read_safe': True, 'version': app.config.version}

    return metadata


if __name__ == '__main__':
    package = sys.argv[1]
    outdir = sys.argv[2]
    try:
        other_defines = sys.argv[3]
    except IndexError:
        other_defines = True
    else:
        other_defines = other_defines in ('True', 'true', '1')

    generate_api_reference_rst(
        app=None, package=package, outdir=outdir, defines=other_defines
    )
