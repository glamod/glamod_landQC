"""Tests for the current values-only SYNOP/METAR API."""
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import pytest

import message


SYNOP = ("SYN13058457 04963 /2301 10142 20098 30099 40151 52001 60002 "
         "700// 333 10173 20127 3/014 5000/ 54405 55300 59022 60001 "
         "70000 90720 91104=")
METAR = "MET058METAR ZSHC 261730Z 12002MPS 9999 FEW033 15/11 Q1014 NOSIG="
VARIABLES = ("temperature", "dew_point_temperature", "sea_level_pressure",
             "wind_speed", "wind_gust")
OBSERVED = pd.Timestamp("2024-05-26 17:30")


@pytest.fixture(autouse=True)
def disable_decoder_prompts(monkeypatch):
    """Keep the decoder's current debug input prompts noninteractive in tests."""
    monkeypatch.setattr("builtins.input", lambda *args: "")


def test_supplied_synop_values():
    expected = {"temperature": 14.2, "dew_point_temperature": 9.8,
                "station_level_pressure": 1009.9, "sea_level_pressure": 1015.1,
                "maximum_temperature": 17.3, "minimum_temperature": 12.7,
                "wind_direction": 230, "wind_speed": np.nan, "wind_gust": np.nan}
    values = message.decode_values(SYNOP, tuple(expected), None, "SYNOP")
    assert set(values) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(values[name], value, equal_nan=True)


def test_supplied_metar_values():
    expected = {"temperature": 15, "dew_point_temperature": 11,
                "wind_speed": 2, "wind_direction": 120, "altimeter": 1014,
                "sky_cover_layer_baseht_1": 1005.84, "sea_level_pressure": np.nan,
                "wind_gust": np.nan, "visibility": np.nan, "sky_cover_layer_1": np.nan}
    values = message.decode_values(METAR, tuple(expected), OBSERVED, "METAR")
    assert set(values) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(values[name], value, equal_nan=True)


def test_alignment_and_no_mutation():
    frame = pd.DataFrame({"REM": [METAR, None, SYNOP, "other", 23],
                          "DATE": [OBSERVED, OBSERVED, None, OBSERVED, OBSERVED]},
                         index=[7, 3, 7, 2, 1])
    original = frame.copy(deep=True)
    values = message.decode_remarks(frame, ["temperature", "wind_speed"])
    np.testing.assert_allclose(values["temperature"],
                               [15, np.nan, 14.2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(values["wind_speed"],
                               [2, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame(index=[3, 3]),
                                   pd.DataFrame({"REM": [None, "", pd.NA, np.nan]})])
def test_missing_remarks(frame):
    values = message.decode_remarks(frame, ["temperature"])
    assert len(values["temperature"]) == len(frame)
    assert np.isnan(values["temperature"]).all()


def test_mcc_decodes_once_and_preserves_flags():
    frame = pd.DataFrame({"REM": [METAR, SYNOP, "other", None],
                          "DATE": [OBSERVED, None, OBSERVED, OBSERVED]})
    station = SimpleNamespace(temperature=SimpleNamespace(flags=np.array(["a", "", "", ""])),
                              wind_speed=SimpleNamespace(flags=np.array(["", "b", "", ""])))
    variables = ["temperature", "wind_speed"]
    with patch.object(message, "decode_remarks", wraps=message.decode_remarks) as decode:
        with patch.object(message, "decode_values", wraps=message.decode_values) as parse:
            with patch.object(message, "message_cross_check") as compare:
                message.mcc(station, frame, variables, {})
    decode.assert_called_once_with(frame, variables)
    assert parse.call_count == 2
    assert compare.call_count == 2
    np.testing.assert_allclose(compare.call_args_list[0].args[3],
                               [15, 14.2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(station.temperature.flags, ["a", "", "", ""])
    np.testing.assert_array_equal(station.wind_speed.flags, ["", "b", "", ""])


@pytest.mark.parametrize("indicator, factor", [(0, 1), (1, 1),
                                               (3, 1852 / 3600), (4, 1852 / 3600)])
def test_synop_wind_units_and_extended_speed(indicator, factor):
    report = f"AAXX 2612{indicator} 58457 04963 /2399 00120 11042 21018 39999 49999 333 91140"
    values = message.decode_values(report, VARIABLES, OBSERVED, "SYNOP")
    assert values["wind_speed"] == pytest.approx(120 * factor)
    assert values["wind_gust"] == pytest.approx(40 * factor)
    assert values["temperature"] == pytest.approx(-4.2)
    assert values["sea_level_pressure"] == pytest.approx(999.9)


@pytest.mark.parametrize("wind, speed, gust", [("12010G20KT", 10 * 1852 / 3600, 20 * 1852 / 3600),
                                                ("12036G72KMH", 10, 20), ("VRB02MPS", 2, np.nan),
                                                ("00000MPS", 0, np.nan), ("12002", np.nan, np.nan),
                                                ("120P99KT", np.nan, np.nan)])
def test_metar_wind_units(wind, speed, gust):
    report = f"METAR ZSHC 261730Z {wind} 9999 M02/M04 Q1014"
    values = message.decode_values(report, VARIABLES + ("wind_direction",), OBSERVED, "METAR")
    np.testing.assert_allclose(values["wind_speed"], speed, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(values["wind_gust"], gust, rtol=1e-6, equal_nan=True)
    assert values["temperature"] == -2
    if wind.startswith("VRB"):
        assert np.isnan(values["wind_direction"])


def test_metar_precision_pressure_precipitation_and_layers():
    report = ("SPECI KEWR 111851Z VRB03G19KT 2SM R04R/3000VP6000FT TSRA BR "
              "FEW015 BKN040CB OVC200 22/22 A2987 RMK AO2 SLP114 P0013 T02270215")
    variables = VARIABLES + ("precipitation_1_hour", "sky_cover_layer_3")
    values = message.decode_values(report, variables, pd.Timestamp("2024-05-11"), "METAR")
    assert values["temperature"] == pytest.approx(22.7)
    assert values["dew_point_temperature"] == pytest.approx(21.5)
    assert values["sea_level_pressure"] == pytest.approx(1011.4)
    assert values["precipitation_1_hour"] == pytest.approx(3.302)
    assert values["sky_cover_layer_3"] == 8


def test_trace_precipitation_is_not_exact_zero():
    report = "METAR ZSHC 261730Z 12002MPS 9999 15/11 Q1014 RMK P0000"
    values = message.decode_values(report, ("temperature", "precipitation_1_hour"),
                                   OBSERVED, "METAR")
    assert values["temperature"] == 15
    assert np.isnan(values["precipitation_1_hour"])


@pytest.mark.parametrize("report, message_type, observed", [
    ("AAXX 26121 58457 NIL", "SYNOP", None),
    ("METAR ZSHC 261730Z NIL", "METAR", OBSERVED),
    ("AAXX 26121 58457", "SYNOP", None),
    ("SYN123not-a-report", "SYNOP", None),
    ("METAR ZSHC 991730Z 12002MPS 9999 15/11 Q1014", "METAR", OBSERVED),
    ("METAR nonsense", "METAR", OBSERVED),
    ("AAXX 26121 58457 04963 /2301 10142 20098 555 12345", "SYNOP", None),
    ("METAR ZSHC 261730Z 12002MPS 9999 15/11 Q1014 RMK UNKNOWN", "METAR", OBSERVED),
    (METAR + " " + METAR, "METAR", OBSERVED),
])
def test_unclear_reports_are_skipped(report, message_type, observed):
    assert message.decode_values(report, VARIABLES, observed, message_type) == {}


def test_date_context_and_invalid_context():
    with patch.object(message.SYNOP, "decode", return_value={}) as decode:
        message.decode_values(SYNOP, ("temperature",), OBSERVED, "SYNOP")
    assert decode.call_args.args[0].startswith("AAXX 2617/")
    assert message.decode_values(METAR, VARIABLES, "invalid", "METAR") == {}
    frame = pd.DataFrame({"REM": [METAR, METAR], "DATE": ["invalid", "2024-05-26"]})
    np.testing.assert_allclose(message.decode_remarks(frame, ["temperature"])["temperature"],
                               [np.nan, 15], equal_nan=True)


def test_repeated_gusts_do_not_select_an_arbitrary_period():
    report = "AAXX 26121 58457 04963 /2301 10142 20098 333 91004 91105"
    values = message.decode_values(report, VARIABLES, OBSERVED, "SYNOP")
    assert values["temperature"] == 14.2
    assert np.isnan(values["wind_gust"])


def test_parser_warnings_skip_row_without_leaking():
    from pymetdecoder import DecodeWarning

    def warn_on_decode(report):
        warnings.warn("Unclear report", DecodeWarning)

    with warnings.catch_warnings(record=True) as caught:
        with patch.object(message.SYNOP, "decode", side_effect=warn_on_decode):
            frame = pd.DataFrame({"REM": [SYNOP, METAR], "DATE": [None, OBSERVED]})
            values = message.decode_remarks(frame, ["temperature"])
    np.testing.assert_allclose(values["temperature"], [np.nan, 15], equal_nan=True)
    assert not caught


def test_programming_errors_are_not_swallowed():
    with patch.object(message.SYNOP, "decode", side_effect=RuntimeError("unexpected")):
        with pytest.raises(RuntimeError, match="unexpected"):
            message.decode_values(SYNOP, VARIABLES, None, "SYNOP")


def test_requested_variables_and_array_storage():
    frame = pd.DataFrame({"REM": [METAR, SYNOP] * 10,
                          "DATE": [OBSERVED, None] * 10})
    values = message.decode_remarks(frame, VARIABLES)
    assert set(values) == set(VARIABLES)
    assert all(isinstance(value, np.ndarray) and value.dtype == np.float64
               for value in values.values())
    assert sum(value.nbytes for value in values.values()) == len(frame) * len(VARIABLES) * 8
    with patch.object(message, "decode_values") as decode:
        assert message.decode_remarks(frame, []) == {}
    decode.assert_not_called()
    with pytest.raises(ValueError, match="Unsupported numeric"):
        message.check_message_variables(["typo"])