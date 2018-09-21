"""Fetcher based on dipy."""

import os
import sys
import contextlib

from os.path import join as pjoin
from hashlib import md5
from shutil import copyfileobj

import tarfile
import zipfile

if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

# Set a user-writeable file-system location to put files:
if 'FURY_HOME' in os.environ:
    fury_home = os.environ['FURY_HOME']
else:
    fury_home = pjoin(os.path.expanduser('~'), '.fury')

# The URL to the University of Washington Researchworks repository:
UW_RW_URL = \
  "https://digital.lib.washington.edu/researchworks/bitstream/handle/"


class FetcherError(Exception):
    pass


def update_progressbar(progress, total_length):
    """Show progressbar
    Takes a number between 0 and 1 to indicate progress from 0 to 100%.
    """
    # Try to set the bar_length according to the console size
    try:
        columns = os.popen('tput cols', 'r').read()
        bar_length = int(columns) - 46
        if(not (bar_length > 1)):
            bar_length = 20
    except Exception:
        # Default value if determination of console size fails
        bar_length = 20
    block = int(round(bar_length * progress))
    size_string = "{0:.2f} MB".format(float(total_length) / (1024 * 1024))
    text = "\rDownload Progress: [{0}] {1:.2f}%  of {2}".format(
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
    """
    Prints a message indicating that a certain data-set is already in place
    """
    msg = 'Dataset is already in place. If you want to fetch it again '
    msg += 'please first remove the folder %s ' % folder
    print(msg)


def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5=None):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5
    Input
    -----
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against. If None (default), checking is
        skipped
    """
    if stored_md5 is not None:
        computed_md5 = _get_file_md5(filename)
        if stored_md5 != computed_md5:
            msg = """The downloaded file, %s, does not have the expected md5
   checksum of "%s". Instead, the md5 checksum was: "%s". This could mean that
   something is wrong with the file or that the upstream file has been updated.
   You can try downloading the file again or updating to the newest version of
   Fury.""" % (filename, stored_md5,
                computed_md5)
            raise FetcherError(msg)


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        try:
            response_size = opener.headers['content-length']
        except KeyError:
            response_size = None

        with open(fname, 'wb') as data:
            if(response_size is None):
                copyfileobj(opener, data)
            else:
                copyfileobj_withprogress(opener, data, response_size)


def fetch_data(files, folder, data_size=None):
    """Downloads files to folder and checks their md5 checksums
    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, md5). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the md5 checksum does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.
    Raises
    ------
    FetcherError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.
    """
    if not os.path.exists(folder):
        print("Creating new folder %s" % (folder))
        os.makedirs(folder)

    if data_size is not None:
        print('Data size is approximately %s' % data_size)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        print('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        check_md5(fullpath, md5)
    if all_skip:
        _already_there_msg(folder)
    else:
        print("Files successfully downloaded to %s" % (folder))


def _make_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                  md5_list=None, doc="", data_size=None, msg=None,
                  unzip=False):
    """ Create a new fetcher
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
    md5_list : list of strings, optional
        The md5 checksums of the files. Used to verify the content of the
        files. Default: None, skipping checking md5.
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
    returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs
    """
    def fetcher():
        files = {}
        for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (baseurl + f, md5_list[i] if
                        md5_list is not None else None)
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


fetch_viz_icons = _make_fetcher("fetch_viz_icons",
                                pjoin(fury_home, "icons"),
                                UW_RW_URL + "1773/38478/",
                                ['icomoon.tar.gz'],
                                ['icomoon.tar.gz'],
                                ['94a07cba06b4136b6687396426f1e380'],
                                data_size="12KB",
                                doc="Download icons for fury",
                                unzip=True)


def read_viz_icons(style='icomoon', fname='infinity.png'):
    """ Read specific icon from specific style
    Parameters
    ----------
    style : str
        Current icon style. Default is icomoon.
    fname : str
        Filename of icon. This should be found in folder HOME/.fury/style/.
        Default is infinity.png.
    Returns
    --------
    path : str
        Complete path of icon.
    """

    folder = pjoin(fury_home, 'icons', style)
    return pjoin(folder, fname)
