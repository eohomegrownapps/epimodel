"""Region database (continents, countries, provinces, GLEAM basins) - codes, names, basic stats, tree structure (TODO).

This module loads region data in the format of
https://github.com/epidemics/epimodel-covid-data/blob/master/regions.csv.
Each region has an ISO-based code (as detailed in :class:`.RegionDataset`);
all datasets are organized by those codes (as a row index).
"""

import datetime
import enum
import logging
import re
import weakref
from collections import OrderedDict
from pathlib import Path

import dateutil
import numpy as np
import pandas as pd
import unidecode

from .utils import normalize_name

log = logging.getLogger(__name__)


class Level(enum.Enum):
    """
    Region levels in the dataset. The numbers are NOT canonical, only the
    names are. These are ordered by "size". From smallest to largest, these are:
    
    - ``Level.gleam_basin``
    - ``Level.subdivision``
    - ``Level.country``
    - ``Level.subregion``
    - ``Level.continent``
    - ``Level.world``
    """

    gleam_basin = 1
    subdivision = 2
    country = 3
    subregion = 4
    continent = 5
    world = 6

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Region:
    """
    A region in the :class:`.RegionDataset`.

    Parameters
    ----------
    rds : epimodel.regions.RegionDataset
        The RegionDataset from which to initialise this region
    code : str
        The unique code identifying this region.

    Attributes
    ----------
    Code : str
        A unique code identifying this region (the code for this region at the smallest available region level).
    continent : str
        The prefixed ISO continent code for this region (or None if not applicable)
    subregion : str
        TBD: The subregion code for this region (or None if not applicable)
    country : str
        The ISOa2 country code for this region (or None if not applicable)
    subdivision : str
        The ISO 3166-2 state/province code for this region (or None if not applicable)
    parent : epimodel.regions.Region
        The region one level above enclosing this region (or None if not applicable)
    children : Set[epimodel.regions.Region]
        The regions one level below enclosed by this region (empty if not applicable)
    """
    
    def __init__(self, rds, code):
        self._rds = weakref.ref(rds)
        self._code = code
        self._parent = None
        self._children = set()
        r = rds.data.loc[code]
        names = [r.Name, r.OfficialName]
        if not pd.isnull(r.OtherNames):
            names.extend(r.OtherNames.split(RegionDataset.SEP))
        names = [n for n in names if not pd.isnull(n) and n]
        rds.data.at[code, "AllNames"] = list(set(names))
        rds.data.at[code, "Region"] = self
        rds.data.at[code, "DisplayName"] = self.get_display_name()

    def get_display_name(self):
        """
        Get the display name of the region.

        Returns
        -------
        str
        """
        if self.Level == Level.subdivision:
            return f"{self.Name}, {self.CountryCode}"
        if self.Level == Level.gleam_basin:
            return f"{self.Name}, {self.SubdivisionCode}"
        return self.Name

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._code} {self.Name} ({self.Level})>"

    def __setattr__(self, name, val):
        """Forbid direct writes to anything but _variables."""
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise AttributeError(
                f"Setting attribute {name} on {self!r} not allowed (use rds.data directly)."
            )

    @property
    def Code(self):
        return self._code

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def _region_prop(self, name):
        """Returns the Region corresponding to code in `self[name]` (None if that is None)."""
        rds = self._rds()
        assert rds is not None
        cid = rds.data.at[self._code, name]
        if pd.isnull(cid) or cid == "":
            return None
        return rds[cid]

    @property
    def continent(self):
        return self._region_prop("ContinentCode")

    @property
    def subregion(self):
        return self._region_prop("SubregionCode")

    @property
    def country(self):
        return self._region_prop("CountryCode")

    @property
    def subdivision(self):
        return self._region_prop("SubdivisionCode")


class RegionDataset:
    """
    A set of regions and their attributes, with a hierarchy. A common index
    for most data files.

    Notes
    -----
    Codes are defined as follows:
    
    - ``W``: The world, root node, Level="world"
    - ``W-AS``: Prefixed ISO continent code, Level="continent"
    - (TBD): Subregion code, Level="subregion"
    - ``US``: ISOa2 code, Level="country"
    - ``US-CA``: ISO 3166-2 state/province code, Level="subdivision"
    - ``G-AAA``: Prefixed IANA code, used for GLEAM basins, Level="gleam_basin"
    
    Attributes
    ----------
    regions : Iterator[epimodel.regions.Region]
        All regions contained in the RegionDataset
    data : pandas.DataFrame
        Index:
            Unique region code
        Columns:
            - Name: Name, dtype: object
            - Name: OfficialName, dtype: object
            - Name: OtherNames, dtype: object
            - Name: Level, dtype: object
            - Name: M49Code, dtype: object
            - Name: ContinentCode, dtype: object
            - Name: SubregionCode, dtype: object
            - Name: CountryCode, dtype: object
            - Name: CountryCodeISOa3, dtype: object
            - Name: SubdivisionCode, dtype: object
            - Name: Lat, dtype: float32
            - Name: Lon, dtype: float32
            - Name: Population, dtype: float32
            - Name: GleamID, dtype: object
            - Name: AllNames, dtype: object
            - Name: Region, dtype: object
            - Name: DisplayName, dtype: object
    """

    # Separating names in name list and column name from date
    SEP = "|"

    LEVELS = pd.CategoricalDtype(pd.Index(list(Level), dtype="O",), ordered=True,)

    COLUMN_TYPES = OrderedDict(
        #        Parent="string",
        # ASCII name (unidecoded)
        Name="U",
        # Official name (any charactersscript)
        OfficialName="U",
        # OtherNames, incl orig. name unicode if different
        # encoded as '|'-separated list
        OtherNames="U",
        # Administrative level
        Level=LEVELS,
        # Countries and above
        M49Code="U",
        # Location in hierarchy
        ContinentCode="U",
        SubregionCode="U",
        CountryCode="U",
        CountryCodeISOa3="U",
        SubdivisionCode="U",
        # Other data
        Lat="f4",
        Lon="f4",
        Population="f4",
        # Stored as string to allow undefined values
        GleamID="U",
    )

    def __init__(self):
        # Main DataFrame (empty)
        self.data = pd.DataFrame(
            index=pd.Index([], name="Code", dtype=pd.StringDtype())
        )
        for name, dtype in self.COLUMN_TYPES.items():
            self.data[name] = pd.Series(dtype=dtype, name=name)
        # name: [Region, Region, ..]
        self._name_index = {}
        # code: [Region, Region, ...]
        self._code_index = {}

    @classmethod
    def load(cls, *paths):
        """
        Creates a RegionDataset and its Regions from the given CSV.

        Optionally also loads other CSVs with additional regions (e.g. GLEAM
        regions)
        
        Parameters
        ----------
        *paths 
            Variable length argument list of CSV paths to load.

        Returns
        -------
        epimodel.regions.RegionDataset
        """
        s = cls()
        cols = dict(cls.COLUMN_TYPES, Level="U")
        for path in paths:
            log.debug("Loading regions from {path!r} ...")
            data = pd.read_csv(
                path,
                dtype=cols,
                index_col="Code",
                na_values=[""],
                keep_default_na=False,
            )
            # Convert Level to enum
            data.Level = data.Level.map(lambda name: Level[name])
            s.data = s.data.append(data)
        s.data.sort_index()
        s._rebuild_index()
        return s

    @property
    def regions(self):
        """Iterator over all regions."""
        return self._code_index.values()

    def __getitem__(self, code):
        """
        Returns the Region corresponding to code, or raise KeyError.
        
        Parameters
        ----------
        code : str
            The code to lookup

        Returns
        -------
        epimodel.regions.Region

        Raises
        ------
        KeyError
            - If region not found
        """
        return self._code_index[code.upper()]

    def get(self, code, default=None):
        """
        Returns the Region corresponding to a given code, or `default`.

        Parameters
        ----------
        code : str
            The code to lookup (e.g. GB)
        default : str, optional
            The Region to return if no such Region is found.
        
        Returns
        -------
        epimodel.regions.Region
        """
        try:
            return self[code]
        except KeyError:
            return default

    def find_all_by_name(self, s, levels=None):
        """
        Returns all Regions with some matching names (filtering on levels).
        
        Parameters
        ----------
        s : str
            The name to lookup (e.g. Australia)
        levels : List[epimodel.regions.Level]
            The levels to filter by. Only Regions within the given levels will be returned.
        
        Returns
        -------
        Tuple[epimodel.regions.Region]
        """
        if levels is not None:
            if isinstance(levels, Level):
                levels = [levels]
            assert all(isinstance(x, Level) for x in levels)
        rs = tuple(self._name_index.get(normalize_name(s), []))
        if levels is not None:
            rs = tuple(r for r in rs if r.Level in levels)
        return rs

    def find_one_by_name(self, s, levels=None):
        """
        Find one region matching name (filter on levels).
        
        Parameters
        ----------
        s : str
            The name to lookup (e.g. Australia)
        levels : List[epimodel.regions.Level]
            The levels to filter by. Only Regions within the given levels will be returned.

        Returns
        -------
        epimodel.regions.Region

        Raises
        ------
        KeyError
            - If no or multiple regions found.
        """
        rs = self.find_all_by_name(s, levels=levels)
        if len(rs) == 1:
            return rs[0]
        lcmt = "" if levels is None else f" [levels={levels!r}]"
        if len(rs) < 1:
            raise KeyError(f"Found no regions matching {s!r}{lcmt}")
        raise KeyError(f"Found multiple regions matching {s!r}{lcmt}: {rs!r}")

    def write_csv(self, path):
        """
        Exports the RegionDataset to CSV.

        Parameters
        ----------
        path : str
            The path to save the CSV to
        """
        # Reconstruct the OtherNames column
        for r in self.regions:
            names = set(r.AllNames)
            if r.Name in names:
                names.remove(r.Name)
            if r.OfficialName in names:
                names.remove(r.OfficialName)
            self.data.loc[r.Code, "OtherNames"] = self.SEP.join(names)
        # Write only non-generated columns
        df = self.data[self.COLUMN_TYPES.keys()]
        # Convert Level to names
        df.Level = df.Level.map(lambda l: l.name)
        # Write
        df.to_csv(path, index_label="Code")

    def _rebuild_index(self):
        """Rebuilds the indexes and ALL Region objects!"""
        self._name_index = {}
        self._code_index = {}
        self.data = self.data.sort_index()
        self.data["AllNames"] = pd.Series(dtype=object)
        self.data["Region"] = pd.Series(dtype=object)
        self.data["DisplayName"] = pd.Series(dtype=object)
        conflicts = []

        # Create Regions
        for ri in self.data.index:
            reg = Region(self, ri)
            for n in set(normalize_name(name) for name in reg.AllNames):
                self._name_index.setdefault(n, list()).append(reg)
            assert ri not in self._code_index
            self._code_index[ri] = reg

        # Unify names in index and find conflicts
        for k in self._name_index:
            self._name_index[k] = list(set(self._name_index[k]))
            if len(self._name_index[k]) > 1:
                conflicts.append(k)
        if conflicts:
            log.info(
                f"Name index has {len(conflicts)} potential conflicts: {conflicts!r}"
            )

        # Add parent/children relations
        for r in self.regions:
            parent = None
            if parent is None and r.Level <= Level.gleam_basin:
                parent = r.subdivision
            if parent is None and r.Level <= Level.subdivision:
                parent = r.country
            if parent is None and r.Level <= Level.country:
                parent = r.subregion
            if parent is None and r.Level <= Level.subregion:
                parent = r.continent
            if parent is None and r.Level < Level.world:
                parent = self.get("W", None)
            r._parent = parent
            if parent is not None:
                parent._children.add(r)
