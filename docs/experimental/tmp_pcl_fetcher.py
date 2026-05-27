from fury.data.fetcher import fury_home
from os.path import join as pjoin


def read_viz_dataset(fname):
    """Read specific dataset.

    Parameters
    ----------
    fname : str
        Filename of the dataset.
        This should be found in folder HOME/.fury/datasets/.

    Returns
    --------
    path : str
        Complete path of dataset.

    """
    folder = pjoin(fury_home, 'datasets')
    return pjoin(folder, fname)
