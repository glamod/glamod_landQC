"""Decode clear SYNOP/METAR values for message cross checks.

Only requested numeric values are returned. REM remains the original source;
parser output is discarded after each row. Comparison/flagging is not yet
implemented.
"""
import re
import warnings
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
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
METAR_ALIASES = {"temperature": "temp", "dew_point_temperature": "dewpt",
                 "sea_level_pressure": "press_sea_level", "altimeter": "press",
                 "wind_direction": "wind_dir", "wind_speed": "wind_speed",
                 "wind_gust": "wind_gust", "visibility": "vis",
                 "snow_depth": "snowdepth"}
WIND_FACTORS = {"KT": 1852 / 3600, "m/s": 1, "MPS": 1, "KMH": 1 / 3.6}


def _variables(var_list):
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


def _synop_value(decoded, var):
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


def _metar_value(parsed, var, wind_units_known, report):
    """Convert one exact METAR quantity to repository units; omit ambiguity."""
    if var in ("wind_speed", "wind_gust") and not wind_units_known:
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


def _decode_values(text, variables, observation_time, message_type):
    """Parse one report and retain only clear requested scalars.

    An empty mapping means the report was skipped. Missing dependencies and
    programming errors propagate; only input conversion and parser failures
    are handled here. No report or diagnostic record is retained.
    """
    report = _normalize(text)
    if not report or "NIL" in report.split():
        return {}
    try:
        observed = None if observation_time is None or pd.isna(observation_time) else pd.Timestamp(observation_time)
    except (ValueError, TypeError):
        return {}
    if observed is not None and pd.isna(observed):
        return {}
    if message_type == "SYNOP":
        from pymetdecoder import DecodeError, DecodeWarning
        from pymetdecoder.synop import SYNOP

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
        if any(decoded.get(key) for key in ("section5", "_not_implemented", "_error")):
            return {}
        return {var: _synop_value(decoded, var) for var in variables}

    from metar import Metar

    if not re.match(r"^(?:METAR|SPECI)\s+(?:COR\s+)?[A-Z][A-Z0-9]{3}\s+\d{6}Z\b", report):
        return {}
    # Use a deterministic parser-only date; no dates are returned as observations.
    observed = observed if observed is not None else pd.Timestamp("2000-01-01")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            parsed = Metar.Metar(report, month=observed.month, year=observed.year, strict=True)
        except (Metar.ParserError, RuntimeWarning):
            return {}
    if parsed.time is None or parsed._unparsed_groups or parsed._unparsed_remarks:
        return {}
    wind_units_known = False
    if "wind_speed" in variables or "wind_gust" in variables:
        for token in report.split(" RMK", 1)[0].split():
            group = Metar.WIND_RE.match(token + " ")
            if group is not None:
                wind_units_known = group.groupdict()["units"] is not None
                break
    return {var: _metar_value(parsed, var, wind_units_known, report) for var in variables}


def decode_synop(text: str, var_list, observation_time=None) -> dict:
    """Return requested numeric values from one raw/SYNnnn-wrapped SYNOP.

    :param str text: land report, with optional AAXX header and terminator
    :param var_list: aliases from SCALAR_FIELDS
    :param observation_time: optional datetime-like day/hour context
    :returns: alias-to-float mapping; NaN for ambiguous fields, empty for a
        rejected report. Units are Celsius, hPa, m/s, degrees, metres or mm.
    :raises ValueError: for unsupported variable names
    """
    return _decode_values(text, _variables(var_list), observation_time, "SYNOP")


def decode_metar(text: str, var_list, observation_time=None) -> dict:
    """Return requested numeric values from one raw/METnnn-wrapped METAR/SPECI.

    :param str text: report with station and observation time
    :param var_list: aliases from SCALAR_FIELDS
    :param observation_time: optional datetime-like month/year context
    :returns: alias-to-float mapping, or empty for a rejected report. Unknown
        wind units and non-exact values are NaN; QNH never substitutes for SLP.
    :raises ValueError: for unsupported variable names
    """
    return _decode_values(text, _variables(var_list), observation_time, "METAR")


def decode_remarks(station_df: pd.DataFrame, var_list, *, message_type=None) -> dict:
    """Decode once per row directly into preallocated float64 arrays.

    :param pd.DataFrame station_df: observations with optional REM/DATE columns
    :param var_list: requested numeric aliases; only these arrays are allocated
    :param message_type: optional SYNOP/METAR filter for single-type adapters
    :returns: alias-to-array mapping in positional row order, with NaN for
        unprocessed values. No messages or decoded objects are retained.
    :raises ValueError: for unsupported aliases or message type
    """
    variables = _variables(var_list)
    if message_type not in (None, "SYNOP", "METAR"):
        raise ValueError(f"Unsupported message type: {message_type}")
    values = {var: np.full(len(station_df), np.nan) for var in variables}
    if not variables or "REM" not in station_df:
        return values
    times = station_df["DATE"] if "DATE" in station_df else repeat(None)
    # Positional iteration preserves alignment even when index labels repeat.
    for position, (text, observed) in enumerate(zip(station_df["REM"], times)):
        if not isinstance(text, str):
            continue
        stripped = text.lstrip()
        if re.match(r"^(?:SYN\d{3}|AAXX\s)", stripped):
            kind = "SYNOP"
        elif re.match(r"^(?:MET\d{3}|METAR\s|SPECI\s)", stripped):
            kind = "METAR"
        else:
            continue
        if message_type is not None and message_type != kind:
            continue
        for var, value in _decode_values(text, variables, observed, kind).items():
            values[var][position] = value
    return values


def read_synops(synops: pd.DataFrame, var: str) -> np.ndarray:
    """Return one SYNOP numeric array in row order; other rows remain NaN."""
    return decode_remarks(synops, [var], message_type="SYNOP")[var]


def read_metars(metars: pd.DataFrame, var: str) -> np.ndarray:
    """Return one METAR numeric array in row order; other rows remain NaN."""
    return decode_remarks(metars, [var], message_type="METAR")[var]


def process_remarks(station_df: pd.DataFrame, var: str, *, decoded=None) -> np.ndarray:
    """Return one numeric array, optionally reusing already decoded arrays.

    :param pd.DataFrame station_df: source rows; REM/DATE are not modified
    :param str var: supported numeric alias
    :param decoded: optional alias-to-array mapping from these same rows
    :returns: row-aligned float array, with NaN for unavailable values
    :raises ValueError: for unsupported aliases or missing/misaligned arrays
    """
    _variables([var])
    if decoded is None:
        return decode_remarks(station_df, [var])[var]
    if var not in decoded or np.shape(decoded[var]) != (len(station_df),):
        raise ValueError("Decoded arrays must contain the variable and match the dataframe length")
    return decoded[var]


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
    values = decode_remarks(station_df, var_list)
    for var in var_list:
        message_cross_check(getattr(station, var), station, station_df, values[var],
                            tsplots=tsplots, diagnostics=diagnostics)