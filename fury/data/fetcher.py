"""Fetcher based on dipy."""

import asyncio
import contextlib
from hashlib import sha256
import json
import os
from os.path import dirname, join as pjoin
import platform
from shutil import copyfileobj
import sys
import tarfile
from urllib.request import urlopen
import warnings
import zipfile

import aiohttp

# Set a user-writeable file-system location to put files:
if 'FURY_HOME' in os.environ:
    fury_home = os.environ['FURY_HOME']
else:
    fury_home = pjoin(os.path.expanduser('~'), '.fury')

# The URL to the University of Washington Researchworks repository:
UW_RW_URL = \
    "https://digital.lib.washington.edu/researchworks/bitstream/handle/"

NEW_ICONS_DATA_URL = (
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/icons/"
    "new_icons/"
)

CUBEMAP_DATA_URL = \
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/cubemaps/"

FURY_DATA_URL = \
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/examples/"

MODEL_DATA_URL = \
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/models/"

TEXTURE_DATA_URL = \
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/textures/"

DMRI_DATA_URL = \
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/dmri/"

GLTF_DATA_URL = \
    "https://api.github.com/repos/KhronosGroup/glTF-Sample-Models/contents/2.0/"  # noqa


class FetcherError(Exception):
    pass


def update_progressbar(progress, total_length):
    """Show progressbar.

    Takes a number between 0 and 1 to indicate progress from 0 to 100%.
    """
    # Try to set the bar_length according to the console size
    try:
        if os.name == 'nt':
            bar_length = 20
        else:
            columns = os.popen('tput cols', 'r').read()
            bar_length = int(columns) - 46
        if bar_length < 1:
            bar_length = 20
    except Exception:
        # Default value if determination of console size fails
        bar_length = 20
    block = int(round(bar_length * progress))
    size_string = "{0:.2f} MB".format(float(total_length) / (1024 * 1024))
    text = "\rDownload Progress: [{0}] {1:.2f}%  of {2}\n".format(
        "#" * block + "-" * (bar_length - block), progress * 100, size_string)
    sys.stdout.write(text)
    sys.stdout.flush()


def copyfileobj_withprogress(fsrc, fdst, total_length, length=16 * 1024):
    copied = 0
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        copied += len(buf)
        progress = float(copied) / float(total_length)
        update_progressbar(progress, total_length)


def _already_there_msg(folder):
    """Print a message indicating that dataset is already in place."""
    msg = 'Dataset is already in place. If you want to fetch it again '
    msg += 'please first remove the folder %s ' % folder
    print(msg)


def _get_file_sha(filename):
    """Generate SHA checksum for the entire file in blocks of 256.

    Parameters
    ----------
    filename : str
        The path to the file whose sha checksum is to be generated

    Returns
    -------
    sha256_data : str
        The computed sha hash from the input file

    """
    sha256_data = sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(256*sha256_data.block_size), b''):
            sha256_data.update(chunk)
    return sha256_data.hexdigest()


def check_sha(filename, stored_sha256=None):
    """Check the generated sha checksum.

    Parameters
    ----------
    filename : str
        The path to the file whose checksum is to be compared
    stored_sha256 : str, optional
        Used to verify the generated SHA checksum.
        Default: None, checking is skipped

    """
    if stored_sha256 is not None:
        computed_sha256 = _get_file_sha(filename)
        if stored_sha256.lower() != computed_sha256:
            msg = """The downloaded file, %s,
             does not have the expected sha
            checksum of "%s".
             Instead, the sha checksum was: "%s".
             This could mean that
            something is wrong with the file
             or that the upstream file has been updated.
            You can try downloading the file again
             or updating to the newest version of
            Fury.""" % (filename, stored_sha256, computed_sha256)
            raise FetcherError(msg)


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        try:
            response_size = opener.headers['content-length']
        except KeyError:
            response_size = None

        with open(fname, 'wb') as data:
            if response_size is None:
                copyfileobj(opener, data)
            else:
                copyfileobj_withprogress(opener, data, response_size)


def fetch_data(files, folder, data_size=None):
    """Download files to folder and checks their sha checksums.

    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, sha). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the sha checksum does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.

    Raises
    ------
    FetcherError

        Raises if the sha checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.

    """
    if not os.path.exists(folder):
        print("Creating new folder %s" % (folder))
        os.makedirs(folder)

    if data_size is not None:
        print('Data size is approximately %s' % data_size)

    all_skip = True
    for f in files:
        url, sha = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and \
           (_get_file_sha(fullpath) == sha.lower()):
            continue
        all_skip = False
        print('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        check_sha(fullpath, sha)
    if all_skip:
        _already_there_msg(folder)
    else:
        print("Files successfully downloaded to %s" % (folder))


def _make_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                  sha_list=None, doc="", data_size=None, msg=None,
                  unzip=False):
    """Create a new fetcher.

    Parameters
    ----------
    name : str
        The name of the fetcher function.
    folder : str
        The full path to the folder in which the files would be placed locally.
        Typically, this is something like 'pjoin(fury_home, 'foo')'
    baseurl : str
        The URL from which this fetcher reads files
    remote_fnames : list of strings
        The names of the files in the baseurl location
    local_fnames : list of strings
        The names of the files to be saved on the local filesystem
    sha_list : list of strings, optional
        The sha checksums of the files. Used to verify the content of the
        files. Default: None, skipping checking sha.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional.
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.

    Returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs

    """
    def fetcher():
        files = {}
        for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (baseurl + f, sha_list[i] if
                        sha_list is not None else None)
        fetch_data(files, folder, data_size)

        if msg is not None:
            print(msg)
        if unzip:
            for f in local_fnames:
                split_ext = os.path.splitext(f)
                if split_ext[-1] == '.gz' or split_ext[-1] == '.bz2':
                    if os.path.splitext(split_ext[0])[-1] == '.tar':
                        ar = tarfile.open(pjoin(folder, f))
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError('File extension is not recognized')
                elif split_ext[-1] == '.zip':
                    z = zipfile.ZipFile(pjoin(folder, f), 'r')
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError('File extension is not recognized')

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


async def _request(session, url):
    """Get the request data as json.

    Parameters
    ----------
    session : ClientSession
        Aiohttp client session.
    url : string
        The URL from which _request gets the response

    Returns
    -------
    response : dictionary
        The response of url request.

    """
    async with session.get(url) as response:
        if not response.status == 200:
            raise aiohttp.InvalidURL(url)

        return await response.json()


async def _download(session, url, filename, size=None):
    """Download file from url.

    Parameters
    ----------
    session : ClientSession
        Aiohttp client session
    url : string
        The URL of the downloadable file
    filename : string
        Name of the downloaded file (e.g. BoxTextured.gltf)
    size : int, optional
        Length of the content in bytes
    """
    if not os.path.exists(filename):
        print(f'Downloading: {filename}')
        async with session.get(url) as response:
            size = response.content_length if not size else size
            block = size
            copied = 0
            with open(filename, mode='wb') as f:
                async for chunk in response.content.iter_chunked(block):
                    f.write(chunk)
                    copied += len(chunk)
                    progress = float(copied)/float(size)
                    update_progressbar(progress, size)


async def _fetch_gltf(name, mode):
    """Fetch glTF samples.

    Parameters
    ----------
    name: str, list
        Name of the glTF model (for e.g. Box, BoxTextured, FlightHelmet, etc)

    mode: str
        Type of the glTF format.
        (e.g. glTF, glTF-Embedded, glTF-Binary, glTF-Draco)

    Returns
    -------
    f_names : list
        list of fetched all file names.
    folder : str
        Path to the fetched files.

    """
    if name is None:
        name = ['BoxTextured', 'Duck', 'CesiumMilkTruck', 'CesiumMan']

    if isinstance(name, list):
        f_names = await asyncio.gather(
            *[_fetch_gltf(element, mode) for element in name]
        )
        return f_names
    else:
        path = f'{name}/{mode}'
        DATA_DIR = pjoin(dirname(__file__), 'files')
        with open(pjoin(DATA_DIR, 'KhronosGltfSamples.json'), 'r') as f:
            models = json.loads(f.read())

        urls = models.get(path, None)

        if urls is None:
            raise ValueError(
                "Model name and mode combination doesn't exist")

        path = pjoin(name, mode)
        path = pjoin('glTF', path)
        folder = pjoin(fury_home, path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        d_urls = [file['download_url'] for file in urls]
        sizes = [file['size'] for file in urls]
        f_names = [url.split('/')[-1] for url in d_urls]
        f_paths = [pjoin(folder, name) for name in f_names]
        zip_url = zip(d_urls, f_paths, sizes)

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[_download(session, url, name, s) for url, name, s in zip_url]
            )

        return f_names, folder


def fetch_gltf(name=None, mode='glTF'):
    """Download glTF samples from Khronos Group Github.

    Parameters
    ----------
    name: str, list, optional
        Name of the glTF model (for e.g. Box, BoxTextured, FlightHelmet, etc)
        https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0
        Default: None, Downloads essential glTF samples for tests.

    mode: str, optional
        Type of glTF format.
        You can choose from different options
        (e.g. glTF, glTF-Embedded, glTF-Binary, glTF-Draco)
        Default: glTF, `.bin` and texture files are stored separately.

    Returns
    -------
    filenames : tuple
        tuple of feteched filenames (list) and folder (str) path.

    """
    if platform.system().lower() == "windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    filenames = asyncio.run(_fetch_gltf(name, mode))
    return filenames


fetch_viz_cubemaps = _make_fetcher(
    "fetch_viz_cubemaps",
    pjoin(fury_home, "cubemaps"),
    CUBEMAP_DATA_URL,
    ['skybox-nx.jpg', 'skybox-ny.jpg', 'skybox-nz.jpg', 'skybox-px.jpg',
     'skybox-py.jpg', 'skybox-pz.jpg'],
    ['skybox-nx.jpg', 'skybox-ny.jpg', 'skybox-nz.jpg', 'skybox-px.jpg',
     'skybox-py.jpg', 'skybox-pz.jpg'],
    ['12B1CE6C91AA3AAF258A8A5944DF739A6C1CC76E89D4D7119D1F795A30FC1BF2',
     'E18FE2206B63D3DF2C879F5E0B9937A61D99734B6C43AC288226C58D2418D23E',
     '00DDDD1B715D5877AF2A74C014FF6E47891F07435B471D213CD0673A8C47F2B2',
     'BF20ACD6817C9E7073E485BBE2D2CE56DACFF73C021C2B613BA072BA2DF2B754',
     '16F0D692AF0B80E46929D8D8A7E596123C76729CC5EB7DFD1C9184B115DD143A',
     'B850B5E882889DF26BE9289D7C25BA30524B37E56BC2075B968A83197AD977F3'],
    doc="Download cube map textures for fury"
)

fetch_viz_icons = _make_fetcher(
    "fetch_viz_icons",
    pjoin(fury_home, "icons"),
    UW_RW_URL + "1773/38478/",
    ['icomoon.tar.gz'],
    ['icomoon.tar.gz'],
    ['BC1FEEA6F58BA3601D6A0B029EB8DFC5F352E21F2A16BA41099A96AA3F5A4735'],
    data_size="12KB",
    doc="Download icons for fury",
    unzip=True
    )

fetch_viz_new_icons = _make_fetcher(
    "fetch_viz_new_icons",
    pjoin(fury_home, "icons", "new_icons"),
    NEW_ICONS_DATA_URL,
    ["circle-pressed.png", "circle.png", "delete-pressed.png", "delete.png",
     "drawing-pressed.png", "drawing.png", "line-pressed.png", "line.png",
     "polyline-pressed.png", "polyline.png", "quad-pressed.png", "quad.png",
     "resize-pressed.png", "resize.png", "selection-pressed.png",
     "selection.png"],
    ["circle-pressed.png", "circle.png", "delete-pressed.png", "delete.png",
     "drawing-pressed.png", "drawing.png", "line-pressed.png", "line.png",
     "polyline-pressed.png", "polyline.png", "quad-pressed.png", "quad.png",
     "resize-pressed.png", "resize.png", "selection-pressed.png",
     "selection.png"],
    ["CD859F244DF1BA719C65C869C3FAF6B8563ABF82F457730ADBFBD7CA72DDB7BC",
     "5896BDC9FF9B3D1054134D7D9A854677CE9FA4E64F494F156BB2E3F0E863F207",
     "937C46C25BC38B62021B01C97A4EE3CDE5F7C8C4A6D0DB75BF4E4CACE2AF1226",
     "476E00A0A5373E1CCDA4AF8E7C9158E0AC9B46B540CE410C6EA47D97F364A0CD",
     "08A914C5DC7997CB944B8C5FBB958951F80B715CFE04FF4F47A73F9D08C4B14B",
     "FB2210B0393ECA8A5DD2B8F034DAE386BBB47EB95BB1CAC2A97DE807EE195ADF",
     "8D1AC2BB7C5BAA34E68578DAAD85F64EF824BE7BCB828CAC18E52833D4CBF4C9",
     "E6D833B6D958129E12FF0F6087282CE92CD43C6DAFCE03F185746ECCA89E42A9",
     "CFF12B8DE48FC19DA5D5F0EA7FF2D23DD942D05468E19522E7C7BEB72F0FF66E",
     "7AFE65EBAE0C0D0556393B979148AE15FC3E037D126CD1DA4A296F4E25F5B4AA",
     "5FD43F1C2D37BF9AF05D9FC591172684AC51BA236980CD1B0795B0225B9247E2",
     "A2DA0CB963401C174919E1D8028AA6F0CB260A736FD26421DB5AB08E9F3C4FDF",
     "FF49DDF9DF24729F4F6345C30C88DE0A11E5B12B2F2FF28375EF9762FE5F8995",
     "A2D850CDBA8F332DA9CD7B7C9459CBDA587C18AF0D3C12CA68D6E6A864EF54BB",
     "54618FDC4589F0A039D531C07A110ED9BC57A256BB15A3B5429CF60E950887C3",
     "CD573F5E4BF4A91A3B21F6124A95FFB3C036F926F8FEC1FD0180F5D27D8F48C0"],
    doc="Download the new icons for DrawPanel"
    )


fetch_viz_wiki_nw = _make_fetcher(
    "fetch_viz_wiki_nw",
    pjoin(fury_home, "examples", "wiki_nw"),
    FURY_DATA_URL,
    ['wiki_categories.txt', 'wiki_edges.txt',
     'wiki_positions.txt'],
    ['wiki_categories.txt', 'wiki_edges.txt',
     'wiki_positions.txt'],
    ['1679241B13D2FD01209160F0C186E14AB55855478300B713D5369C12854CFF82',
     '702EE8713994243C8619A29C9ECE32F95305737F583B747C307500F3EC4A6B56',
     '044917A8FBD0EB980D93B6C406A577BEA416FA934E897C26C87E91C218EF4432'],
    doc="Download the following wiki information"
        "Interdisciplinary map of the journals",
    msg=("More information about complex "
         "networks can be found in this papers:"
         " https://arxiv.org/abs/0711.3199")
    )

fetch_viz_models = _make_fetcher(
    "fetch_viz_models",
    pjoin(fury_home, "models"),
    MODEL_DATA_URL,
    ['utah.obj', 'suzanne.obj', 'satellite_obj.obj', 'dragon.obj'],
    ['utah.obj', 'suzanne.obj', 'satellite_obj.obj', 'dragon.obj'],
    ['0B50F12CEDCDC27377AC702B1EE331223BECEC59593B3F00A9E06B57A9C1B7C3',
     'BB4FF4E65D65D71D53000E06D2DC7BF89B702223657C1F64748811A3A6C8D621',
     '90213FAC81D89BBB59FA541643304E0D95C2D446157ACE044D46F259454C0E74',
     'A775D6160D04EAB9A4E90180104F148927CEFCCAF9F0BCD748265CB8EE86F41B'],
    doc=" Download the models for shader tutorial"
    )

fetch_viz_dmri = _make_fetcher(
    "fetch_viz_dmri",
    pjoin(fury_home, "dmri"),
    DMRI_DATA_URL,
    ['fodf.nii.gz', 'slice_evecs.nii.gz', 'slice_evals.nii.gz',
     'roi_evecs.nii.gz', 'roi_evals.nii.gz', 'whole_brain_evecs.nii.gz',
     'whole_brain_evals.nii.gz'],
    ['fodf.nii.gz', 'slice_evecs.nii.gz', 'slice_evals.nii.gz',
     'roi_evecs.nii.gz', 'roi_evals.nii.gz', 'whole_brain_evecs.nii.gz',
     'whole_brain_evals.nii.gz'],
    ['767ca3e4cd296e78421d83c32201b30be2d859c332210812140caac1b93d492b',
     '8843ECF3224CB8E3315B7251D1E303409A17D7137D3498A8833853C4603C6CC2',
     '3096B190B1146DD0EADDFECC0B4FBBD901F4933692ADD46A83F637F28B22122D',
     '89E569858A897E72C852A8F05BBCE0B21C1CA726E55508087A2DA5A38C212A17',
     'F53C68CCCABF97F1326E93840A8B5CE2E767D66D692FFD955CA747FFF14EC781',
     '8A894F6AB404240E65451FA6D10FB5D74E2D0BDCB2A56AD6BEA38215BF787248',
     '47A73BBE68196381ED790F80F48E46AC07B699B506973FFA45A95A33023C7A77']
)

fetch_viz_textures = _make_fetcher(
    "fetch_viz_textures",
    pjoin(fury_home, "textures"),
    TEXTURE_DATA_URL,
    ['1_earth_8k.jpg', '2_no_clouds_8k.jpg',
     '5_night_8k.jpg', 'earth.ppm',
     'jupiter.jpg', 'masonry.bmp',
     'moon_8k.jpg',
     '8k_mercury.jpg', '8k_venus_surface.jpg',
     '8k_mars.jpg', '8k_saturn.jpg',
     '8k_saturn_ring_alpha.png',
     '2k_uranus.jpg', '2k_neptune.jpg',
     '8k_sun.jpg', '1_earth_16k.jpg',
     'clouds.jpg'],
    ['1_earth_8k.jpg', '2_no_clouds_8k.jpg',
     '5_night_8k.jpg', 'earth.ppm',
     'jupiter.jpg', 'masonry.bmp',
     'moon-8k.jpg',
     '8k_mercury.jpg', '8k_venus_surface.jpg',
     '8k_mars.jpg', '8k_saturn.jpg',
     '8k_saturn_ring_alpha.png',
     '2k_uranus.jpg', '2k_neptune.jpg',
     '8k_sun.jpg', '1_earth_16k.jpg',
     'clouds.jpg'],
    ['0D66DC62768C43D763D3288CE67128AAED27715B11B0529162DC4117F710E26F',
     '5CF740C72287AF7B3ACCF080C3951944ADCB1617083B918537D08CBD9F2C5465',
     'DF443F3E20C7724803690A350D9F6FDB36AD8EBC011B0345FB519A8B321F680A',
     '34CE9AD183D7C7B11E2F682D7EBB84C803E661BE09E01ADB887175AE60C58156',
     '5DF6A384E407BD0D5F18176B7DB96AAE1EEA3CFCFE570DDCE0D34B4F0E493668',
     '045E30B2ABFEAE6318C2CF955040C4A37E6DE595ACE809CE6766D397C0EE205D',
     '7397A6C2CE0348E148C66EBEFE078467DDB9D0370FF5E63434D0451477624839',
     '5C8BD885AE3571C6BA2CD34B3446B9C6D767E314BF0EE8C1D5C147CADD388FC3',
     '9BC21A50577ED8AC734CDA91058724C7A741C19427AA276224CE349351432C5B',
     '4CC52149924ABC6AE507D63032F994E1D42A55CB82C09E002D1A567FF66C23EE',
     '0D39A4A490C87C3EDABE00A3881A29BB3418364178C79C534FE0986E97E09853',
     'F1F826933C9FF87D64ECF0518D6256B8ED990B003722794F67E96E3D2B876AE4',
     'D15239D46F82D3EA13D2B260B5B29B2A382F42F2916DAE0694D0387B1204A09D',
     'CB42EA82709741D28B0AF44D8B283CBC6DBD0C521A7F0E1E1E010ADE00977DF6',
     'F22B1CFB306DDCE72A7E3B628668A0175B745038CE6268557CB2F7F1BDF98B9D',
     '7DD1DAC926101B5D7B7F2E952E53ACF209421B5CCE57C03168BCE0AAD675998A',
     '85043336E023C4C9394CFD6D48D257A5564B4F895BFCEC01C70E4898CC77F003'],
    doc="Download textures for fury"
    )


def read_viz_cubemap(name, suffix_type=1, ext='.jpg'):
    """Read specific cube map with specific suffix type and extension.

    Parameters
    ----------
    name : str
    suffix_type : int, optional
        0 for numeric suffix (e.g., skybox_0.jpg, skybox_1.jpg, etc.), 1 for
        -p/nC encoding where C is either x, y or z (e.g., skybox-px.jpeg,
        skybox-ny.jpeg, etc.), 2 for pos/negC where C is either x, y, z (e.g.,
        skybox_posx.png, skybox_negy.png, etc.), and 3 for position in the cube
        map (e.g., skybox_right.jpg, skybox_front.jpg, etc).
    ext : str, optional
        Image type extension. (.jpg, .jpeg, .png, etc.).

    Returns
    -------
    list of paths : list
        List with the complete paths of the skybox textures.

    """
    # Set of commonly used cube map naming conventions and its associated
    # indexing number. For a correct creation and display of the skybox,
    # textures must be read in this order.
    suffix_types = {
        0: ['0', '1', '2', '3', '4', '5'],
        1: ['-px', '-nx', '-py', '-ny', '-pz', '-nz'],
        2: ['posx', 'negx', 'posy', 'negy', 'posz', 'negz'],
        3: ['right', 'left', 'top', 'bottom', 'front', 'back']
    }
    if suffix_type in suffix_types:
        conv = suffix_types[suffix_type]
    else:
        warnings.warn('read_viz_cubemap(): Invalid suffix_type.')
        return None
    cubemap_fnames = []
    folder = pjoin(fury_home, 'cubemaps')
    for dir_conv in conv:
        cubemap_fnames.append(pjoin(folder, name + dir_conv + ext))
    return cubemap_fnames


def read_viz_icons(style='icomoon', fname='infinity.png'):
    """Read specific icon from specific style.

    Parameters
    ----------
    style : str
        Current icon style. Default is icomoon.
    fname : str
        Filename of icon. This should be found in folder HOME/.fury/style/.
        Default is infinity.png.

    Returns
    -------
    path : str
        Complete path of icon.

    """
    if not os.path.isdir(pjoin(fury_home, 'icons', style)):
        if style == "icomoon":
            fetch_viz_icons()
        elif style == "new_icons":
            fetch_viz_new_icons()
    folder = pjoin(fury_home, 'icons', style)
    return pjoin(folder, fname)


def read_viz_models(fname):
    """Read specific model.

    Parameters
    ----------
    fname : str
        Filename of the model.
        This should be found in folder HOME/.fury/models/.

    Returns
    -------
    path : str
        Complete path of models.

    """
    folder = pjoin(fury_home, 'models')
    return pjoin(folder, fname)


def read_viz_textures(fname):
    """Read specific texture.

    Parameters
    ----------
    fname: str
        Filename of the texture.
        This should be found in folder HOME/.fury/textures/.

    Returns
    -------
    path : str
        Complete path of textures.

    """
    folder = pjoin(fury_home, 'textures')
    return pjoin(folder, fname)


def read_viz_dmri(fname):
    """Read specific dMRI image.

    Parameters
    ----------
    fname: str
        Filename of the texture.
        This should be found in folder HOME/.fury/dmri/.

    Returns
    -------
    path : str
        Complete path of dMRI image.

    """
    folder = pjoin(fury_home, 'dmri')
    return pjoin(folder, fname)


def read_viz_gltf(fname, mode='glTF'):
    """Read specific gltf sample.

    Parameters
    ----------
    fname : str
        Name of the model.
        This should be found in folder HOME/.fury/models/glTF/.

    mode : str, optional
        Model type (e.g. glTF-Binary, glTF-Embedded, etc)
        Default : glTF

    Returns
    -------
    path : str
        Complete path of models.

    """
    folder = pjoin(fury_home, 'glTF')
    model = pjoin(folder, fname)

    sample = pjoin(model, mode)

    if not os.path.exists(sample):
        raise ValueError(f'Model {sample} does not exists.')

    for filename in os.listdir(sample):
        if filename.endswith('.gltf') or filename.endswith('.glb'):
            return pjoin(sample, filename)


def list_gltf_sample_models():
    """Return all model name from the glTF-samples repository.

    Returns
    -------
    model_names : list
        Lists the name of glTF sample from
        https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0

    """
    DATA_DIR = pjoin(dirname(__file__), 'files')
    with open(pjoin(DATA_DIR, 'KhronosGltfSamples.json'), 'r') as f:
        models = json.loads(f.read())
    models = models.keys()
    model_modes = [model.split('/')[0] for model in models]

    model_names = []
    for name in model_modes:
        if name not in model_names:
            model_names.append(name)
    model_names = model_names[1:]  # removing __comments__

    default_models = ['BoxTextured', 'Duck', 'CesiumMilkTruck', 'CesiumMan']

    if not model_names:
        print('Failed to get models list')
        return None
    result = [model in model_names for model in default_models]
    for i, exist in enumerate(result):
        if not exist:
            print(f'Default Model: {default_models[i]} not found!')
    return model_names
