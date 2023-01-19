from pathlib import Path
import os
import shutil

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def abort(error):
    print(f'*WARNING* Examples Revamp not generated: \n\n{error}')
    exit()


def gallery_order():
    pass


def prepare_gallery(app=None):
    examples_dir = os.path.join(app.srcdir, '..', 'examples')
    examples_revamp_dir = os.path.join(app.srcdir, '..', 'examples_revamp')
    os.makedirs(examples_revamp_dir, exist_ok=True)

    f_example_desc = Path(examples_dir, "_valid_examples.toml")
    if not f_example_desc.exists():
        msg = "No valid examples description file found "
        msg += "(e.g '_valid_examples.toml')"
        abort(msg)

    with open(f_example_desc, 'rb') as fobj:
        # import ipdb; ipdb.set_trace()
        desc_examples = tomllib.load(fobj)

    import ipdb; ipdb.set_trace()
    all_files_found = []
    all_files_not_found = []
    all_files_not_included = []

    print(desc_examples)
    main_section = desc_examples.get('main')
    all_files_found = all_files_found + main_section.get('files', [])

    for f in main_section.get('files'):
        if not Path(examples_dir, f).exists():
            msg = f"File {f} not found in {examples_dir}"
            all_files_not_found.append(f)
        shutil.copy(Path(examples_dir, f), Path(examples_revamp_dir, f))




    print(dir(app))

    #examples found
    #examples not found
    #examples not selected


def setup(app):
    """Install the plugin.

    Parameters
    ----------
    app: Sphinx application context.

    """
    from sphinx.util import logging
    logger = logging.getLogger(__name__)
    logger.info('Initializing Examples folder revamp plugin...')

    app.connect('builder-inited', prepare_gallery)
    # app.connect('build-finished', summarize_failing_examples)

    metadata = {'parallel_read_safe': True,
                'version': app.config.version}

    return metadata


if __name__ == '__main__':
    gallery_name = sys.argv[1]
    outdir = sys.argv[2]

    prepare_gallery(app=None, package=package, outdir=outdir)
