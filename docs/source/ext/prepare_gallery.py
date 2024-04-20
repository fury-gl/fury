from dataclasses import dataclass, field
import fnmatch
import os
from pathlib import Path
import shutil

from sphinx.util import logging

try:
    import tomllib
except ImportError:
    import tomli as tomllib

logger = logging.getLogger(__name__)


@dataclass(order=True)
class examplesConfig:
    sort_index: int = field(init=False)
    readme: str
    position: int
    enable: bool
    files: list
    folder_name: str

    def __post_init__(self):
        self.sort_index = self.position


def abort(error):
    print(f'*WARNING* Examples Revamp not generated: \n\n{error}')
    exit()


def prepare_gallery(app=None):
    examples_dir = os.path.join(app.srcdir, '..', 'examples')
    examples_revamp_dir = os.path.join(app.srcdir, '..', 'examples_revamped')
    os.makedirs(examples_revamp_dir, exist_ok=True)

    f_example_desc = Path(examples_dir, '_valid_examples.toml')
    if not f_example_desc.exists():
        msg = 'No valid examples description file found '
        msg += "(e.g '_valid_examples.toml')"
        abort(msg)

    with open(f_example_desc, 'rb') as fobj:
        try:
            desc_examples = tomllib.load(fobj)
        except Exception as e:
            msg = f'Error Loading examples description file: {e}.\n\n'
            msg += 'Please check the file format.'
            abort(msg)

    if 'main' not in desc_examples.keys():
        msg = 'No main section found in examples description file'
        abort(msg)

    try:
        examples_config = sorted(
            [examplesConfig(folder_name=k, **v) for k, v in desc_examples.items()]
        )
    except Exception as e:
        msg = f'Error parsing examples description file: {e}.\n\n'
        msg += 'Please check the file format.'
        abort(msg)

    if examples_config[0].position != 0:
        msg = 'Main section must be first in examples description file with position=0'
        abort(msg)
    elif examples_config[0].folder_name != 'main':
        msg = "Main section must be named 'main' in examples description file"
        abort(msg)
    elif examples_config[0].enable is False:
        msg = 'Main section must be enabled in examples description file'
        abort(msg)

    disable_examples_section = []

    for example in examples_config:
        if not example.enable:
            disable_examples_section.append(example.folder_name)
            continue

        # Create folder for each example
        if example.position != 0:
            folder = Path(
                examples_revamp_dir, f'{example.position:02d}_{example.folder_name}'
            )
        else:
            folder = Path(examples_revamp_dir)

        if not folder.exists():
            os.makedirs(folder)

        # Create readme file
        if example.readme.startswith('file:'):
            filename = example.readme.split('file:')[1].strip()
            shutil.copy(Path(examples_dir, filename), Path(folder, 'README.rst'))
        else:
            with open(Path(folder, 'README.rst'), 'w') as fi:
                fi.write(example.readme)

        # Copy files to folder
        if not example.files:
            continue

        for fi in example.files:
            if not Path(examples_dir, fi).exists():
                msg = f'\tFile {fi} not found in examples folder: {examples_dir}.\n\n'
                msg += '\tPlease, Add the file or remove it from the description file.'
                logger.info(msg)
                continue
            shutil.copy(Path(examples_dir, fi), Path(folder, fi))

    # Check if all python examples are in the description file
    files_in_config = [fi for ex in examples_config for fi in ex.files]
    all_examples = fnmatch.filter(os.listdir(examples_dir), '*.py')
    for all_ex in all_examples:
        if all_ex in files_in_config:
            continue
        msg = f'File {all_ex} not found in examples description file: {f_example_desc}'
        logger.info(msg)


def setup(app):
    """Install the plugin.

    Parameters
    ----------
    app: Sphinx application context.

    """
    logger.info('Initializing Examples folder revamp plugin...')

    app.connect('builder-inited', prepare_gallery)
    # app.connect('build-finished', summarize_failing_examples)

    metadata = {'parallel_read_safe': True, 'version': app.config.version}

    return metadata


if __name__ == '__main__':
    gallery_name = sys.argv[1]
    outdir = sys.argv[2]

    prepare_gallery(app=None, package=package, outdir=outdir)
