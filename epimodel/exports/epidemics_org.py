import datetime
import getpass
import json
import logging
import socket
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..regions import Region

log = logging.getLogger(__name__)

MAIN_DATA_FILENAME = "data-CHANNEL-v4.json"


class WebExport:
    """
    CHANGED

    Document holding one data export to web. Contains a subset of Regions.
 
    Attributes
    ----------
    created : datetime.datetime
        Set to the current UTC time
    created_by : str
        Set to ``{getpass.getuser()}@{socket.gethostname()}``
    comment : str
        An optional comment (default: `None`)
    export_regions : dict[str, epimodel.exports.epidemics_org.WebExportRegion]
        The Regions to export
    
    Parameters
    ----------
    comment : str, optional
        An optional comment (default: `None`)
    """

    def __init__(self, date_resample: str, comment=None):
        self.created = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.created_by = f"{getpass.getuser()}@{socket.gethostname()}"
        self.comment = comment
        self.date_resample = date_resample
        self.export_regions = {}

    def to_json(self):
        """
        Converts all regions in `self.export_regions` to JSON and returns a 
        `dict` containing `WebExport` data.
        
        Returns a dict with the following structure
        
        ::
            {
                "created": <self.created>,
                "created_by": <self.created_by>,
                "comment": <self.comment>,
                "regions": <regions>
            }
        
        <regions>: A dict mapping region codes to JSON-encoded WebExportRegion data (``WebExportRegion.to_JSON()``)
        """
        return {
            "created": self.created,
            "created_by": self.created_by,
            "comment": self.comment,
            "date_resample": self.date_resample,
            "regions": {k: a.to_json() for k, a in self.export_regions.items()},
        }

    def new_region(
        self,
        region,
        models: pd.DataFrame,
        simulation_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: Optional[pd.DataFrame],
        un_age_dist: Optional[pd.DataFrame],
        traces_v3: Optional[pd.DataFrame],
    ):
        """
        CHANGED

        Adds a region to the `WebExport`, and returns the region as a
        `WebExportRegion` object.
        
        Parameters
        ----------
        region : epimodel.regions.Region
        
        Returns
        -------
        epimodel.exports.epidemics_org.WebExportRegion
        """
        er = WebExportRegion(
            region,
            models,
            simulation_spec,
            rates,
            hopkins,
            foretold,
            timezones,
            un_age_dist,
            traces_v3,
        )
        self.export_regions[region.Code] = er
        return er

    def write(self, path, name=None):
        """
        Writes regions added to the object to JSON files in a specified
        directory.
    
        Parameters
        ----------
        path : str
            Path of directory containing the export directory 
        name : str, optional
            Name of export directory to create (if None, a name will be 
            generated based on creation time of the `WebExport` object)
        """
        if name is None:
            name = f"export-{self.created.isoformat()}"
            if self.comment:
                name += self.comment
        name = name.replace(" ", "_").replace(":", "-")
        outdir = Path(path)
        assert (not outdir.exists()) or outdir.is_dir()
        exdir = Path(path) / name
        log.info(f"Writing WebExport to {exdir} ...")
        exdir.mkdir(exist_ok=False, parents=True)
        for rc, er in tqdm(list(self.export_regions.items()), desc="Writing regions"):
            fname = f"extdata-{rc}.json"
            er.data_url = f"{name}/{fname}"
            with open(exdir / fname, "wt") as f:
                json.dump(er.data_ext, f, allow_nan=False)
        with open(exdir / MAIN_DATA_FILENAME, "wt") as f:
            json.dump(self.to_json(), f, default=types_to_json, allow_nan=False)
        log.info(f"Exported {len(self.export_regions)} regions to {exdir}")


class WebExportRegion:
    """
    CHANGED
    
    Stores region data for web export.
    
    Attributes
    ----------
    region : epimodel.regions.Region
    data : dict
        {name: anything}
        Any per-region data. Large ones should go to data_ext.
    data_ext : dict
        {name: anything}
        Extended data to be written in a separate per-region file
    data_url : string
        Relative URL of the extended data file, set on write
    
    Parameters
    ----------
    region : epimodel.regions.Region
    """
    def __init__(
        self,
        region,
        models: pd.DataFrame,
        simulations_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: Optional[pd.DataFrame],
        un_age_dist: Optional[pd.DataFrame],
        traces_v3: Optional[pd.DataFrame],
    ):
        assert isinstance(region, Region)
        self.region = region
        # Any per-region data. Large ones should go to data_ext.
        self.data = self.extract_smallish_data(
            rates, hopkins, foretold, timezones, un_age_dist, traces_v3
        )
        # Extended data to be written in a separate per-region file
        self.data_ext = self.extract_models_data(models, simulations_spec)
        # Relative URL of the extended data file, set on write
        self.data_url = None

    @staticmethod
    def extract_smallish_data(
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: Optional[pd.DataFrame],
        un_age_dist: Optional[pd.DataFrame],
        traces_v3: Optional[pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        """
        CHANGED TODO
        """
        d = {}

        if rates is not None:
            d["Rates"] = rates.replace({np.nan: None}).to_dict()

        if hopkins is not None:
            d["JohnsHopkins"] = {
                "Date": [x.date().isoformat() for x in hopkins.index],
                **hopkins.replace({np.nan: None}).to_dict(orient="list"),
            }

        if foretold is not None:
            d["Foretold"] = {
                "Date": [x.isoformat() for x in foretold.index],
                **foretold.replace({np.nan: None})
                .loc[:, ["Mean", "Variance", "0.05", "0.50", "0.95"]]
                .to_dict(orient="list"),
            }

        if traces_v3 is not None:
            d["TracesV3"] = traces_v3["TracesV3"]

        d["Timezones"] = timezones["Timezone"].tolist()

        if un_age_dist is not None:
            d["AgeDist"] = un_age_dist.to_dict()

        return d

    @staticmethod
    def extract_models_data(
        models: pd.DataFrame, simulation_spec: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        CHANGED TODO
        """
        d = {
            "date_index": [x.isoformat() for x in models.index.levels[0]],
        }
        traces = []
        for simulation_id, simulation_def in simulation_spec.iterrows():
            trace_data = models.xs(simulation_id, level="SimulationID")
            trace = {
                "group": simulation_def["Group"],
                "key": simulation_def["Key"],
                "name": simulation_def["Name"],
                "infected": trace_data.loc[:, "Infected"].tolist(),
                "recovered": trace_data.loc[:, "Recovered"].tolist(),
            }
            traces.append(trace)
        d["traces"] = traces
        return {"models": d}

    def to_json(self):
        """
        Converts `WebExportRegion` to a JSON-friendly dict.
        
        Returns a dict with the following structure:
        
        ::
            {
                "data": <self.data>,
                "data_url": <self.data_url>,
                "Name": <self.region.DisplayName>,
                "Population", <self.region.Population>
                "Lat", ...
                "Lon", ...
                "OfficialName", ...
                "Level", ...
                "M49Code", ...
                "ContinentCode", ...
                "SubregionCode", ...
                "CountryCode", ...
                "CountryCodeISOa3", ...
                "SubdivisionCode", ...
            }

        """

        d = {
            "data": self.data,
            "data_url": self.data_url,
            "Name": self.region.DisplayName,
        }
        for n in [
            "Population",
            "Lat",
            "Lon",
            "OfficialName",
            "Level",
            "M49Code",
            "ContinentCode",
            "SubregionCode",
            "CountryCode",
            "CountryCodeISOa3",
            "SubdivisionCode",
        ]:
            d[n] = None if pd.isnull(self.region[n]) else self.region[n]
        return d


def raise_(msg):
    raise Exception(msg)


def assert_valid_json(file):
    with open(file, "r") as blob:
        json.load(
            blob,
            parse_constant=(lambda x: raise_("Not valid JSON: detected `" + x + "'")),
        )


def upload_export(dir_to_export, gs_prefix, gs_url, channel="test"):
    """
    The 'upload' subcommand (uploads to Google Cloud Storage)
    
    Parameters
    ----------
    dir_to_export : str
        The local directory to upload
    gs_prefix : str
        Google Cloud Storage directory prefix
    gs_url : str
        Google Cloud Storage URL
    channel : str, optional
        Channel to upload to (default: "test")
    """
    CMD = [
        "gsutil",
        "-m",
        "-h",
        "Cache-Control:public,max-age=30",
        "cp",
        "-a",
        "public-read",
    ]
    gs_prefix = gs_prefix.rstrip("/")
    gs_url = gs_url.rstrip("/")
    exdir = Path(dir_to_export)
    assert exdir.is_dir()

    assert_valid_json(exdir / MAIN_DATA_FILENAME)

    log.info(f"Uploading data folder {exdir} to {gs_prefix}/{exdir.parts[-1]} ...")
    cmd = CMD + ["-Z", "-R", exdir, gs_prefix]
    log.debug(f"Running {cmd!r}")
    subprocess.run(cmd, check=True)

    datafile = MAIN_DATA_FILENAME.replace("CHANNEL", channel)
    gs_tgt = f"{gs_prefix}/{datafile}"
    log.info(f"Uploading main data file to {gs_tgt} ...")
    cmd = CMD + ["-Z", exdir / MAIN_DATA_FILENAME, gs_tgt]
    log.debug(f"Running {cmd!r}")
    subprocess.run(cmd, check=True)
    log.info(f"File URL: {gs_url}/{datafile}")

    if channel != "main":
        log.info(f"Custom web URL: http://epidemicforecasting.org/?channel={channel}")


def types_to_json(obj):
    """
    Convert object to a JSON friendly type
    
    Parameters
    ----------
    obj : object
    
    Returns
    -------
    object
    
    Raises
    ------
    TypeError
        If object is unserializable
    """
    if isinstance(obj, (np.float16, np.float32, np.float64, np.float128)):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.name
    else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


def get_cmi(df: pd.DataFrame):
    return df.index.levels[0].unique()


def analyze_data_consistency(
    debug: Optional[None],
    export_regions: List[str],
    models,
    rates_df,
    hopkins,
    foretold,
) -> None:
    codes = {
        "models": get_cmi(models),
        "hopkins": get_cmi(hopkins),
        "foretold": get_cmi(foretold),
        "rates": rates_df.index.unique(),
    }

    union_codes = set()
    any_nan = False
    for source_name, ixs in codes.items():
        if ixs.isna().sum() > 0:
            log.error("Dataset %s contains NaN in index!", source_name)
            any_nan = True
        union_codes.update(ixs)

    if any_nan:
        raise ValueError("Some datasets indexed by NaNs. Fix the source data.")

    df = pd.DataFrame(index=sorted(union_codes))
    for source_name, ixs in codes.items():
        df[source_name] = pd.Series(True, index=ixs)
    df = df.fillna(False)

    log.info("Total data availability, number of locations: %s", df.sum().to_dict())
    log.info("Export requested for %s regions: %s", len(export_regions), export_regions)

    if debug:
        _df = df.loc[export_regions, ["hopkins", "rates"]]
        res = _df.loc[~_df.all(axis=1)].replace({False: "Missing", True: "OK"})
        log.debug(
            "Data presence for hopkins or rates in the following countries: \n%s", res
        )

    diff_export_and_models = set(export_regions).difference(get_cmi(models))
    if diff_export_and_models:
        log.error(
            "You requested to export %s but that's not modelled yet.",
            diff_export_and_models,
        )
        raise ValueError(
            f"Regions {diff_export_and_models} not present in modelled data. Remove it from config."
        )

    log.info(
        "From exported regions (N=%s): %s",
        len(export_regions),
        df.loc[export_regions].sum().to_dict(),
    )


def get_df_else_none(df: pd.DataFrame, code) -> Optional[pd.DataFrame]:
    if code in df.index:
        return df.loc[code].sort_index()
    else:
        return None


def get_df_list(df: pd.DataFrame, code) -> pd.DataFrame:
    return df.loc[[code]].sort_index()


def get_extra_path(args, name: str) -> Path:
    return Path(args.config["data_dir"]) / args.config["web_export"][name]


def aggregate_countries(
    hopkins: pd.DataFrame, mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    to_append = []
    all_state_codes = []
    for country_code, state_codes in mapping.items():
        log.info(
            "Aggregating hopkins data for %s into a single code %s",
            state_codes,
            country_code,
        )
        aggregated = (
            hopkins.loc[state_codes]
            .reset_index("Date")
            .groupby("Date")
            .sum()
            .assign(Code=country_code)
            .reset_index()
            .set_index(["Code", "Date"])
        )
        to_append.append(aggregated)
        all_state_codes.extend(state_codes)
    return hopkins.drop(index=all_state_codes).append(pd.concat(to_append))


def process_export(args) -> None:
    ex = WebExport(args.config["gleam_resample"], comment=args.comment)

    hopkins = get_extra_path(args, "john_hopkins")
    foretold = get_extra_path(args, "foretold")
    rates = get_extra_path(args, "rates")
    timezone = get_extra_path(args, "timezones")
    un_age_dist = get_extra_path(args, "un_age_dist")
    traces_v3 = get_extra_path(args, "traces_v3")

    export_regions = sorted(args.config["export_regions"])

    simulation_specs: pd.DataFrame = pd.read_hdf(args.models_file, "simulations")
    models_df: pd.DataFrame = pd.read_hdf(args.models_file, "new_fraction")

    rates_df: pd.DataFrame = pd.read_csv(rates, index_col="Code", keep_default_na=False)
    timezone_df: pd.DataFrame = pd.read_csv(
        timezone, index_col="Code", keep_default_na=False
    )

    un_age_dist_df: pd.DataFrame = pd.read_csv(un_age_dist, index_col="Code M49").drop(
        columns=["Type", "Region Name", "Parent Code M49"]
    )

    traces_v3_df: pd.DataFrame = pd.read_csv(traces_v3, index_col="CodeISO3")

    hopkins_df: pd.DataFrame = pd.read_csv(
        hopkins, index_col=["Code", "Date"], parse_dates=["Date"], keep_default_na=False,
    ).pipe(aggregate_countries, args.config["state_to_country"])
    foretold_df: pd.DataFrame = pd.read_csv(
        foretold, index_col=["Code", "Date"], parse_dates=["Date"], keep_default_na=False,
    )

    analyze_data_consistency(
        args.debug, export_regions, models_df, rates_df, hopkins_df, foretold_df
    )

    for code in export_regions:
        reg = args.rds[code]
        m49 = int(reg["M49Code"])
        iso3 = reg["CountryCodeISOa3"]
        ex.new_region(
            reg,
            models_df.loc[code].sort_index(level="Date"),
            simulation_specs,
            get_df_else_none(rates_df, code),
            get_df_else_none(hopkins_df, code),
            get_df_else_none(foretold_df, code),
            get_df_list(timezone_df, code),
            get_df_else_none(un_age_dist_df, m49),
            get_df_else_none(traces_v3_df, iso3),
        )

    ex.write(args.config["output_dir"])
