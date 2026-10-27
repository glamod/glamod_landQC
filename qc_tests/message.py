"""
Message checks
==============

Compare values in REMarks section from available SYNOP or METAR messages
to values in data fields.  Flag when these are inconsistent.

Only decodes clear SYNOP/METAR values for message cross checks.

Only requested numeric values are returned. REM remains the original source;
parser output is discarded after each row.
"""
import re
import warnings
from itertools import repeat

from pymetdecoder import DecodeError, DecodeWarning
from pymetdecoder.synop import SYNOP
from metar import Metar

import numpy as np
import pandas as pd

import utils


SCALAR_FIELDS = {
    "temperature", "dew_point_temperature", "station_level_pressure",
    "sea_level_pressure", "wind_speed", "wind_gust", "wind_direction",
    "relative_humidity", "visibility", "altimeter", "snow_depth",
    "wet_bulb_temperature", "maximum_temperature", "minimum_temperature",
    "precipitation_1_hour", "precipitation_3_hour", "precipitation_6_hour",
    "precipitation_24_hour",
} | {f"sky_cover_layer_{layer}" for layer in range(1, 5)} | {
    f"sky_cover_layer_baseht_{layer}" for layer in range(1, 5)
}

SYNOP_ALIASES = {"temperature": "air_temperature",
                 "dew_point_temperature": "dewpoint_temperature",
                 "station_level_pressure": "station_pressure"}

METAR_ALIASES = {"temperature": "temp",
                 "dew_point_temperature": "dewpt",
                 "sea_level_pressure": "press_sea_level",
                 "altimeter": "press",
                 "wind_direction": "wind_dir",
                 "wind_speed": "wind_speed",
                 "wind_gust": "wind_gust",
                 "visibility": "vis",
                 "snow_depth": "snowdepth"}

WIND_FACTORS = {"KT": 1852 / 3600,
                "m/s": 1,
                "MPS": 1,
                "KMH": 1 / 3.6}  # convert wind units to m/s


def check_message_variables(var_list):
    """Validate numeric aliases and retain their order without duplicates."""
    variables = tuple(dict.fromkeys(var_list))
    for var in variables:
        if var not in SCALAR_FIELDS:
            raise ValueError(f"Unsupported numeric message variable: {var}")
    return variables


def _normalize(text):
    """Remove a recognized wrapper/terminator, rejecting concatenated reports."""
    if not isinstance(text, str):
        return ""
    report = " ".join(text.split()).rstrip("=").strip()
    if "=" in report:
        return ""
    return re.sub(r"^(?:SYN|MET)\d{3}", "", report).strip()


def _exact(field):
    """Return a SYNOP quantity as a float, or NaN for non-exact values."""
    if not isinstance(field, dict):
        return np.nan

    if any(field.get(key) for key in ("quantifier", "trace", "varAllUnknown", "obscured")):
        return np.nan

    value = field.get("value")
    return float(value) if isinstance(value, (int, float)) else np.nan


def synop_value(decoded, var):
    """Extract one requested SYNOP value without copying decoded dictionaries."""

    if var in ("wind_speed", "wind_direction", "wind_gust"):
        wind = decoded.get("surface_wind") or {}
        if var == "wind_direction":
            return _exact(wind.get("direction"))

        indicator = decoded.get("wind_indicator") or {}
        if var == "wind_gust":
            gusts = decoded.get("highest_gust") or []
            if len(gusts) != 1:
                return np.nan
            quantity = gusts[0].get("speed") or {}
        else:
            quantity = wind.get("speed") or {}
        factor = WIND_FACTORS.get(quantity.get("unit") or indicator.get("unit"), np.nan)
        return _exact(quantity) * factor

    if var.startswith("sky_cover_layer"):
        layer = int(var[-1]) - 1
        clouds = decoded.get("cloud_layer") or []
        key = "cloud_height" if "baseht" in var else "cloud_cover"

        return _exact(clouds[layer].get(key)) if layer < len(clouds) else np.nan

    if var.startswith("precipitation_"):
        period = int(var.split("_")[1])
        amounts = []
        for section in ("precipitation_s1", "precipitation_s3", "precipitation_24h"):
            precip = decoded.get(section) or {}
            duration = precip.get("time_before_obs") or {}
            if duration.get("unit") == "h" and duration.get("value") == period:
                amounts.append(_exact(precip.get("amount")))
        # Different sections must agree before their shared period is usable.
        return amounts[0] if amounts and all(value == amounts[0] for value in amounts) else np.nan

    return _exact(decoded.get(SYNOP_ALIASES.get(var, var)))


def metar_value(parsed, var, wind_units_known, report):
    """Convert one exact METAR quantity to repository units; omit ambiguity."""

    if var in ("wind_speed", "wind_gust") and not wind_units_known:
        # if unknown wind units, can't parse
        return np.nan

    if var.startswith("sky_cover_layer"):
        layer = int(var[-1]) - 1
        if layer >= len(parsed.sky):
            return np.nan
        cover, height, _ = parsed.sky[layer]
        if "baseht" in var:
            return height.value("M") if height is not None else np.nan

        # Partial cloud categories describe ranges, not exact okta amounts.
        return 8.0 if cover == "OVC" else (0.0 if cover in ("CLR", "SKC", "NSC", "NCD") else np.nan)

    if var.startswith("precipitation_"):
        period = var.split("_")[1]
        quantity = getattr(parsed, f"precip_{period}hr", None)
        if period == "1" and re.search(r"\bP0000\b", report):
            return np.nan
        # python-metar precipitation output supports inches, not millimetres.
        return quantity.value("IN") * 25.4 if quantity is not None else np.nan

    quantity = getattr(parsed, METAR_ALIASES.get(var, ""), None)
    if quantity is None or getattr(quantity, "_gtlt", None):
        return np.nan

    if var == "wind_direction":
        return quantity.value()

    units = {"temperature": "C", "dew_point_temperature": "C",
             "sea_level_pressure": "HPA", "altimeter": "HPA",
             "wind_speed": "MPS", "wind_gust": "MPS", "visibility": "M",
             "snow_depth": "M"}

    return quantity.value(units[var]) if var in units else np.nan


def decode_synop(observed: pd.Timestamp,
                 report: str,
                 variables: tuple) -> dict:
    """Setup SYNOP message for decoding

    Parameters
    ----------
    observed : pd.Timestamp
        Timestamp being processed
    report : str
        SYNOP report string
    variables : tuple
        Variables to extract

    Returns
    -------
    dict
        Extracted values
    """

    # Headerless archives omit YYGGiw. Never infer the missing wind units.
    if re.match(r"^\d{5}\s", report):
        header = observed.strftime("%d%H") + "/" if observed is not None else "/////"
        report = f"AAXX {header} {report}"

    if not re.match(r"^AAXX\s+[\d/]{5}\s+\d{5}\s+[\d/]{5}\s+[\d/]{5}(?:\s|$)", report):
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DecodeWarning)
        try:
            decoded = SYNOP().decode(report)
        except (DecodeError, DecodeWarning):
            return {}

    print(decoded)
    # skip if section5 or errors
    if any(decoded.get(key) for key in ("section5", "_not_implemented", "_error")):
        return {}
    # otherwise decode values normally
    values = {var: synop_value(decoded, var) for var in variables}
    input(f"SYN {values}")
    return values


def decode_metar(observed: pd.Timestamp,
                 report: str,
                 variables: tuple) -> dict:
    """Setup the METAR message for decoding

    Parameters
    ----------
    observed : pd.Timestamp
        Timestamp being processed
    report : str
        METAR report string
    variables : tuple
        Variables to extract

    Returns
    -------
    dict
        Extracted values
    """

    # skip if no METAR header
    if not re.match(r"^(?:METAR|SPECI)\s+(?:COR\s+)?[A-Z][A-Z0-9]{3}\s+\d{6}Z\b", report):
        return {}
    if observed is None:
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            parsed = Metar.Metar(report, month=observed.month, year=observed.year, strict=True)
        except (Metar.ParserError, RuntimeWarning):
            return {}

    print(parsed)
    # skip if parsing failed
    if parsed.time is None or parsed._unparsed_groups or parsed._unparsed_remarks:
        return {}

    # check wind units if needed
    wind_units_known = False
    if "wind_speed" in variables or "wind_gust" in variables:
        for token in report.split(" RMK", 1)[0].split():
            group = Metar.WIND_RE.match(token + " ")
            if group is not None:
                wind_units_known = group.groupdict()["units"] is not None
                break
    # and decode values
    values = {var: metar_value(parsed, var, wind_units_known, report) for var in variables}
    input(f"MET {values}")
    return values

def decode_values(text: str,
                  variables: tuple,
                  observation_time,
                  message_type: str):
    """Parse one report and retain only clear requested scalars.

    An empty mapping means the report was skipped. Missing dependencies and
    programming errors propagate; only input conversion and parser failures
    are handled here. No report or diagnostic record is retained.
    """
    if message_type not in ("SYNOP", "METAR"):
        # can't do other message types (Shouldn't get this far)
        return {}

    report = _normalize(text)
    if not report or "NIL" in report.split():
        return {}

    # if Date doesn't make sense, don't try to parse
    try:
        observed = None if observation_time is None or pd.isna(observation_time) else pd.Timestamp(observation_time)
    except (ValueError, TypeError):
        return {}
    # and check that Timestamp conversion results in valid date
    if observed is not None and pd.isna(observed):
        return {}

    # pass single message to relevant decoder
    if message_type == "SYNOP":
        return decode_synop(observed, report, variables)

    elif message_type == "METAR":
        return decode_metar(observed, report, variables)


def decode_remarks(station_df: pd.DataFrame,
                   var_list: list) -> dict:
    """Decode once per row directly into arrays.

    :param pd.DataFrame station_df: observations with optional REM/DATE columns
    :param var_list: requested numeric aliases; only these arrays are allocated

    :returns: alias-to-array mapping in positional row order, with NaN for
        unprocessed values. No messages or decoded objects are retained.

    :raises ValueError: for unsupported aliases or message type
    """

    # cross check between supplied and supported variables
    variables = check_message_variables(var_list)

    # make array to hold values
    values = {var: np.full(len(station_df), np.nan) for var in variables}
    if not variables or "REM" not in station_df:
        return values

    times = station_df["DATE"] if "DATE" in station_df else repeat(None)

    # Positional iteration preserves alignment even when index labels repeat.
    for position, (text, observed) in enumerate(zip(station_df["REM"], times)):

        if not isinstance(text, str):
            # if no sensible Remarks column value, skip
            continue

        # extract message type from start of text
        stripped = text.lstrip()
        if re.match(r"^(?:SYN\d{3}|AAXX\s)", stripped):
            kind = "SYNOP"
        elif re.match(r"^(?:MET\d{3}|METAR\s|SPECI\s)", stripped):
            kind = "METAR"
        else:
            # and if unknown, then skip on
            continue

        # decode values for this row, and store
        for var, value in decode_values(text, variables, observed, kind).items():
            values[var][position] = value  # dict of arrays

    return values


def message_cross_check(obs_var: "utils.MeteorologicalVariable",
                        station: "utils.Station", station_df: pd.DataFrame,
                        remarks_value: np.ndarray, tsplots: bool = False,
                        diagnostics: bool = False) -> None:
    """Comparison placeholder: leave station values and QC flags unchanged."""
    return


def mcc(station: "utils.Station", station_df: pd.DataFrame,
        var_list: list, config_dict: dict, full: bool = False,
        tsplots: bool = False, diagnostics: bool = False) -> None:
    """Decode requested arrays once and pass them to the comparison stub.

    :param station: station with attributes named by var_list
    :param pd.DataFrame station_df: observations with optional REM/DATE columns
    :param list var_list: numeric aliases to extract
    :param dict config_dict: unused configuration settings
    :param bool full: unused full-update switch
    :param bool tsplots: plotting switch forwarded to the comparison stub
    :param bool diagnostics: diagnostic switch forwarded to the comparison stub

    :returns: None; no observation or QC flag changes
    """

    # initially unpack all the requested variables from the REM column
    values = decode_remarks(station_df, var_list)

    # now do the cross check
    for var in var_list:
        message_cross_check(getattr(station, var), station, station_df, values[var],
                            tsplots=tsplots, diagnostics=diagnostics)