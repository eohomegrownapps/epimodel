"""
Utility functions for handling CSVs.
"""
import re

import dateutil
import pandas as pd
import unidecode


def read_csv(
    path,
    regions=None,
    date_column="Date",
    drop_unknown=True,
    drop_underscored=True,
    **kwargs,
):
    """
    Read given CSV and do some basic checks and create indexes.

    Parameters
    ----------
    path : str
        Path of CSV
    regions : RegionDataset, optional
        If given, checks that CSV codes are in the dataset and warns if they are not.
    date_column : str, optional
        Column to use as secondary index (default is 'Date'). Dates are converted to datetime with UTC timezone.
    drop_unknown : bool, optional
        If `regions` are given, drops rows with unknown codes (default is `True`)
    drop_underscored : bool, optional
        Drops any "_Underscored" columns (including the informative "_Name"). Default is `True`.
    **kwargs
        Passed to `pd.read_csv`

    Returns
    -------
    pandas.DataFrame
    """

    data = pd.read_csv(path, index_col="Code", **kwargs)
    if date_column in data.columns:
        dti = pd.DatetimeIndex(pd.to_datetime(data[date_column], utc=True))
        del data[date_column]
        data.index = pd.MultiIndex.from_arrays([data.index, dti])
    if drop_underscored:
        for n in list(data.columns):
            if n.startswith("_"):
                del data[n]
    data.sort_index()

    # TODO check against regions
    return data


def write_csv(df, path, regions=None, with_name=True):
    """
    Write given CSV normally from a `DataFrame`, adding purely informative "_Name" column by default.
    
    Parameters
    ----------
    df : pandas.DataFrame
    path : str
    with_name : bool, optional
        Add a (purely informative) "_Name" column to the CSV (default is `True`).
    regions : RegionDataset, optional
        The `RegionDataset` to take names from. Required if `with_name` set to `True`.
    """

    if with_name and regions is None:
        raise ValueError("Provide `regions` with `with_name=True`")
    if with_name:
        ns = pd.Series(regions.data.DisplayName, name="_Name")
        df = df.join(ns, how="inner")
    df.write_csv(path)


def normalize_name(name):
    """
    Return normalized version of the name for matching region names.

    Name is unidecoded, lowercased, '-' and '_' are replaced by spaces,
    whitespace is stripped.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    return unidecode.unidecode(name).lower().replace("-", " ").replace("_", " ").strip()
