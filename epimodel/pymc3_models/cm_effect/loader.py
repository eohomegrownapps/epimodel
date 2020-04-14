### Initial imports

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ... import RegionDataset, read_csv

log = logging.getLogger(__name__)


class Loader:
    """
    Loader for countermeasures data.

    Attributes
    ----------
    data_dir : str
        Directory in which data is stored
    Ds : pandas.DateTimeIndex
        Date range from which to extract data
    CMs : list[str]
        List of names of countermeasures (TODO: document all possible countermeasures) (?)
    Rs : list[str]
        List of region codes to use in the model
    rds : epimodel.regions.RegionDataset
        For region data
    johns_hopkins : pandas.DataFrame
        Johns Hopkins data, indexed by `MultiIndex` (code and date) (see :mod:`epimodel.imports.johns_hopkins` for more details)
    features : pandas.DataFrame
        Countermeasures features data, indexed by `MultiIndex` (code and date). Columns 
        are labelled by feature; each cell indicates [(?) whether or not the feature
        is active in the country at that time - TODO: clarify this]
    sel_features : pandas.DataFrame
        The subset of data from `features` that is selected (i.e. regions in
        `Rs`, countermeasures in `CMs` and dates within the date range `Ds`)
    TheanoType : str
        Data type to store numerical data (default: "float64")
    Confirmed : numpy.ma.masked_array
        Masked array containing Johns Hopkins data for confirmed cases
        indexed by region and day (shape is ``(len(Rs), len(Ds))``),
        with all values below `ConfirmedCutoff` marked as invalid / masked.
    ConfirmedCutoff : float
        Cut-off below which values are marked as invalid (default is 10.0)
    Deaths : numpy.ma.masked_array
        Masked array containing Johns Hopkins data for deaths
        indexed by region and day (shape is ``(len(Rs), len(Ds))``),
        with all values below `DeathsCutoff` marked as invalid / masked.
    DeathsCutoff : float
        Cut-off below which values are marked as invalid (default is 10.0)
    Active : numpy.ma.masked_array
        Masked array containing Johns Hopkins data for active cases
        indexed by region and day (shape is ``(len(Rs), len(Ds))``),
        with all values below `ActiveCutoff` marked as invalid / masked.
    ActiveCutoff : float
        Cut-off below which values are marked as invalid (default is 10.0)
    Recovered : numpy.ma.masked_array
        Masked array containing Johns Hopkins data for confirmed cases
        indexed by region and day (shape is ``(len(Rs), len(Ds))``),
        with all values below `RecoveredCutoff` marked as invalid / masked.
    RecoveredCutoff : float
        Cut-off below which values are marked as invalid (default is 10.0)
    ActiveCMs : numpy.ndarray
        Array indexed by region, countermeasure and day, indicating 
        which countermeasures are active and to what extent (shape is 
        ``(len(Rs), len(CMs), len(Ds))``). [Values should be between 0 and 1 
        (?)]
    
    Parameters
    ----------
    start : str or datetime-like
        Left bound for date range
    end : str or datetime-like
        Right bound for date range
    regions : list[str]
        List of region codes to use in the model
    CMs : list[str]
        List of names of countermeasures (TODO: document all possible countermeasures) (?)
    data_dir : str or pathlib.PurePath, optional
        Path of the directory storing CSV data files (``regions.csv``, ``johns-hopkins.csv``
        and the active countermeasures file). Default: ``<parent directory of module>/data``
    active_cm_file : str, optional
        Filename of countermeasure features CSV file. Default: ``countermeasures-model-0to1.csv``

    """
    def __init__(
        self,
        start,
        end,
        regions,
        CMs,
        data_dir=None,
        active_cm_file="countermeasures-model-0to1.csv",
    ):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data"
        self.data_dir = data_dir

        # Days
        self.Ds = pd.date_range(start=start, end=end, tz="utc")

        # CM features
        self.CMs = list(CMs)

        # Countries / regions
        self.Rs = list(regions)

        self.rds = RegionDataset.load(self.data_dir / "regions.csv")

        # Raw data, never modified
        self.johns_hopkins = read_csv(self.data_dir / "johns-hopkins.csv")
        self.features = read_csv(self.data_dir / active_cm_file)

        self.TheanoType = "float64"

        self.Confirmed = None
        self.ConfirmedCutoff = 10.0
        self.Deaths = None
        self.DeathsCutoff = 10.0
        self.Active = None
        self.ActiveCutoff = 10.0
        self.Recovered = None
        self.RecoveredCutoff = 10.0

        self.ActiveCMs = None

        self.update()

    def update(self):
        """(Re)compute the values used in the model after any parameter/region/etc changes."""

        def prep(name, cutoff=None):
            # Confirmed cases, masking values smaller than 10
            v = (
                self.johns_hopkins[name]
                .loc[(tuple(self.Rs), self.Ds)]
                .unstack(1)
                .values
            )
            assert v.shape == (len(self.Rs), len(self.Ds))
            if cutoff is not None:
                v[v < cutoff] = np.nan
            # [country, day]
            return np.ma.masked_invalid(v.astype(self.TheanoType))

        self.Confirmed = prep("Confirmed", self.ConfirmedCutoff)
        self.Deaths = prep("Deaths", self.DeathsCutoff)
        self.Recovered = prep("Recovered", self.RecoveredCutoff)
        self.Active = prep("Active", self.ActiveCutoff)

        self.ActiveCMs = self.get_ActiveCMs(self.Ds[0], self.Ds[-1])

    def get_ActiveCMs(self, start, end):
        """
        Return active countermeasures within a range of dates.

        Parameters
        ----------
        start : str or datetime-like
            Left bound for date range
        end : str or datetime-like
            Right bound for date range

        Returns
        -------
        np.ndarray
            Array indexed by region, countermeasure and day, indicating 
            which countermeasures are active and to what extent (shape is 
            ``(len(Rs), len(CMs), len(Ds))``). [Values should be between 0 and 1 
            (?)]
        """
        local_Ds = pd.date_range(start=start, end=end, tz="utc")
        self.sel_features = self.features.loc[self.Rs, self.CMs]
        if "Mask wearing" in self.sel_features.columns:
            self.sel_features["Mask wearing"] *= 0.01
        ActiveCMs = np.stack(
            [self.sel_features.loc[rc].loc[local_Ds].T for rc in self.Rs]
        )
        assert ActiveCMs.shape == (len(self.Rs), len(self.CMs), len(local_Ds))
        # [region, CM, day] Which CMs are active, and to what extent
        return ActiveCMs.astype(self.TheanoType)

    def print_stats(self):
        """
        Print data stats, plot graphs, ... TODO: add more

        Currently calculates min / mean / max of feature values for each selected
        countermeasure, along with a list of the unique values each countermeasure
        takes (so long as there are <=4 of these).

        Sample output:
        ::
            Countermeasures                            min   .. mean  .. max
             0 Masks over 60                              0.000 .. 0.017 .. 1.000  {0.0, 1.0}
             1 Asymptomatic contact isolation             0.000 .. 0.118 .. 1.000  {0.0, 1.0}
             2 Gatherings limited to 10                   0.000 .. 0.140 .. 1.000  {0.0, 1.0}
             3 Gatherings limited to 100                  0.000 .. 0.214 .. 1.000  {0.0, 1.0}
             4 Gatherings limited to 1000                 0.000 .. 0.259 .. 1.000  {0.0, 1.0}
             5 Business suspended - some                  0.000 .. 0.290 .. 1.000  {0.0, 1.0}
             6 Business suspended - many                  0.000 .. 0.203 .. 1.000  {0.0, 1.0}
             7 Schools and universities closed            0.000 .. 0.366 .. 1.000  {0.0, 1.0}
             8 General curfew - permissive                0.000 .. 0.178 .. 1.000  {0.0, 1.0}
             9 General curfew - strict                    0.000 .. 0.128 .. 1.000  {0.0, 1.0}
            10 Healthcare specialisation over 0.2         0.000 .. 0.059 .. 1.000  {0.0, 1.0}
        """

        print("\nCountermeasures                            min   .. mean  .. max")
        for i, cm in enumerate(self.CMs):
            vals = np.array(self.sel_features[cm])
            print(
                f"{i:2} {cm:42} {vals.min():.3f} .. {vals.mean():.3f}"
                f" .. {vals.max():.3f}"
                f"  {set(vals) if len(set(vals)) <= 4 else ''}"
            )

        # TODO: add more

    def filter_regions(
        self, regions, min_feature_sum=1.0, min_final_jh=400, jh_col="Confirmed"
    ):
        """
        Filter and return list of region codes.
        (Deprecated? Doesn't seem to be used.) (?)
        """
        res = []
        for rc in regions:
            r = self.rds[rc]
            if rc in self.johns_hopkins.index and rc in self.features_0to1.index:
                if self.johns_hopkins.loc[(rc, self.Ds[-1]), jh_col] < min_final_jh:
                    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                    continue
                # TODO: filter by features?
                # if self.active_features.loc[(rc, self.Ds)] ...
                #    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                #    continue
                res.append(rc)
        return res


def split_0to1_features(features_0to1, exclusive=False):
    """
    Split joined features in model-0to1 into separate bool features.

    TODO: Document some of these measures in another page (?)
    
    Parameters
    ----------
    features_0to1 : pandas.DataFrame
    exclusive : bool, optional
        If `exclusive`, only one of a chain of features is activated. 
        Otherwise all up to the active level are active.
        Default: false
    
    Returns
    -------
    pandas.DataFrame
    """
    fs = {}
    f01 = features_0to1

    fs["Masks over 60"] = f01["Mask wearing"] >= 60

    fs["Asymptomatic contact isolation"] = f01["Asymptomatic contact isolation"]

    fs["Gatherings limited to 10"] = f01["Gatherings limited to"] > 0.84
    fs["Gatherings limited to 100"] = f01["Gatherings limited to"] > 0.35
    fs["Gatherings limited to 1000"] = f01["Gatherings limited to"] > 0.05
    if exclusive:
        fs["Gatherings limited to 1000"] &= ~fs["Gatherings limited to 100"]
        fs["Gatherings limited to 100"] &= ~fs["Gatherings limited to 10"]

    fs["Business suspended - some"] = f01["Business suspended"] > 0.1
    fs["Business suspended - many"] = f01["Business suspended"] > 0.6
    if exclusive:
        fs["Business suspended - some"] &= ~fs["Business suspended - many"]

    fs["Schools and universities closed"] = f01["Schools and universities closed"]

    fs["Distancing and hygiene over 0.2"] = (
        f01["Minor distancing and hygiene measures"] > 0.2
    )

    fs["General curfew - permissive"] = f01["General curfew"] > 0.1
    fs["General curfew - strict"] = f01["General curfew"] > 0.6
    if exclusive:
        fs["General curfew - permissive"] &= ~fs["General curfew - strict"]

    fs["Healthcare specialisation over 0.2"] = f01["Healthcare specialisation"] > 0.2

    fs["Phone line"] = f01["Phone line"]

    return pd.DataFrame(fs).astype("f4")
