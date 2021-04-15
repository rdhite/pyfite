# test_coordinates.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Unit tests for pyfite.coordinates

The primary correctness tests are the few for logic added on top of existing conversion libraries such as
approximating degree->meter units at a given latitude.

NOTE: These tests are largely unnecessary due to the fact that actual coordinate conversion math is
done within dependencies (pyproj, pymap3d) that have their own testing for correctness.
These tests ensure those libraries are used properly, numpy array shapes are maintained,
and axis placement is consistent (x, E/W, Longitude are always [0] while y, N/S, Latitude are always [1]).
"""
from cmath import isclose
import unittest

import numpy as np

import pyfite.coordinates as pfc

__METER_TOLERANCE = 0.01
__DEGREE_TOLERANCE = 0.000001


def forward_reverse(crs1: pfc.CoordinateReferenceSystem, crs2: pfc.CoordinateReferenceSystem,  # pylint: disable=too-many-arguments
                    pts1: np.ndarray, pts2: np.ndarray, tol1: float, tol2: float):
    """Tests forward and reverse conversions of points provided two coordinate systems.

    Ensure converting `pts1` via a forward conversion (from `crs1` to `crs2`) yields `pts2` within a tolerance of `tol1`
    Ensure converting `pts2` via a reverse conversion (from `crs2` to `crs1`) yields `pts1` within a tolerance of `tol2`
    """
    forward = pfc.CoordinateConverter(crs1, crs2)
    reverse = pfc.CoordinateConverter(crs2, crs1)

    forward_expected = pts2
    forward_actual = forward(pts1)
    np.testing.assert_array_almost_equal(
        forward_actual, forward_expected, tol1)

    reverse_expected = pts1
    reverse_actual = reverse(pts2)
    np.testing.assert_array_almost_equal(
        reverse_actual, reverse_expected, tol2)


def assert_proj_str_equivalent(proj1: str, proj2: str):
    """Checks that two proj strings are equivalent.

    This presumes parameters are populated with '=' such as '+datum=WGS84'
    instead of '+datum WGS84'
    """
    case = unittest.TestCase()
    case.assertCountEqual(proj1.split(), proj2.split())


# PROJ string generation tests
def test_gdc_get_proj_str():
    """Test that `Geodetic.get_proj_str` returns a correct proj string.
    """
    # no offset
    crs = pfc.Geodetic()
    assert_proj_str_equivalent(
        crs.get_proj_str(), '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    # offset is arbitrary as it does not matter here
    crs = pfc.Geodetic(
        offset=(np.random.rand(), np.random.rand(), np.random.rand()))
    assert_proj_str_equivalent(
        crs.get_proj_str(), '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


def test_gcc_get_proj_str():
    """Test that `Geocentric.get_proj_str` returns a correct proj string.
    """
    # no offset
    crs = pfc.Geocentric()
    assert_proj_str_equivalent(
        crs.get_proj_str(), '+proj=geocent +ellps=WGS84')

    # offset is arbitrary as it does not matter here
    crs = pfc.Geocentric(
        offset=(np.random.rand(), np.random.rand(), np.random.rand()))
    assert_proj_str_equivalent(
        crs.get_proj_str(), '+proj=geocent +ellps=WGS84')


def test_utm_get_proj_str():
    """Test that `Utm.get_proj_str` returns a correct proj string.
    """
    for zone, south in [(3, False), (12, False), (8, True), (17, True), (43, False), (51, True)]:
        crs_no_off = pfc.Utm(zone, south)
        crs_off = pfc.Utm(zone, south, offset=(np.random.rand(),
                                               np.random.rand(), np.random.rand()))
        utm_str = f'+proj=utm +zone={zone} +ellps=WGS84{" +south" if south else ""}'
        assert_proj_str_equivalent(crs_no_off.get_proj_str(), utm_str)
        assert_proj_str_equivalent(crs_off.get_proj_str(), utm_str)


def test_proj_get_proj_str():
    """Test that `ProjCrs.get_proj_str` returns a correct proj string.
    """
    projs = [
        '+proj=merc +lat_ts=56.5 +ellps=GRS80',
        '+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0',
        '+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
    ]

    for proj in projs:
        assert_proj_str_equivalent(pfc.ProjCrs(proj).get_proj_str(), proj)
        assert_proj_str_equivalent(
            pfc.ProjCrs(proj, offset=(np.random.rand(), np.random.rand(),
                                      np.random.rand())).get_proj_str(),
            proj)


def test_offset_application():
    """Test that offsets are applied properly during conversions.
    """
    # If all below test_crs1_crs2 tests pass, we can use them to
    # generate test cases for offset applications

    # Making a round-trip with intermittent checks
    gdc_pts = np.array([[-73.9864461, 40.7567137, 15.000]])
    gcc_pts = pfc.CoordinateConverter(
        pfc.Geodetic(), pfc.Geocentric())(gdc_pts)
    utm_pts = pfc.CoordinateConverter(
        pfc.Geodetic(), pfc.Utm(zone=18, south=False))(gdc_pts)

    # Offsets for each coordinate system
    # Using np.array with shape (1, 3) so subsequent math utilizing offsets is seamless
    gdc_offset = np.array([[-70, 40, 10]])
    gcc_offset = np.array([[1_000_000, -4_000_000, 4_000_000]])
    utm_offset = np.array([[500_000, 4_000_000, 15.000]])

    # Coordinate systems with offsets in place
    off_gdc = pfc.Geodetic(offset=tuple(gdc_offset[0]))
    off_gcc = pfc.Geocentric(offset=tuple(gcc_offset[0]))
    off_utm = pfc.Utm(zone=18, south=False, offset=tuple(utm_offset[0]))

    # Starting spot
    off_gdc_pts_expected = gdc_pts - gdc_offset

    # Checks offset application Geodetic -> Geocentric
    off_gcc_pts_actual = pfc.CoordinateConverter(
        off_gdc, off_gcc)(off_gdc_pts_expected)
    off_gcc_pts_expected = gcc_pts - gcc_offset
    np.testing.assert_array_almost_equal(
        off_gcc_pts_actual, off_gcc_pts_expected, __METER_TOLERANCE)

    # Checks offset application Geocentric -> Utm
    off_utm_pts_actual = pfc.CoordinateConverter(
        off_gcc, off_utm)(off_gcc_pts_expected)
    off_utm_pts_expected = utm_pts - utm_offset
    np.testing.assert_array_almost_equal(
        off_utm_pts_actual, off_utm_pts_expected, __METER_TOLERANCE)

    # Checks offset application Utm -> Geodetic
    off_gdc_pts_actual = pfc.CoordinateConverter(
        off_utm, off_gdc)(off_utm_pts_expected)
    np.testing.assert_array_almost_equal(
        off_gdc_pts_expected[:, :2], off_gdc_pts_actual[:, :2], __DEGREE_TOLERANCE)
    np.testing.assert_array_almost_equal(
        off_gdc_pts_expected[:, 2], off_gdc_pts_actual[:, 2], __METER_TOLERANCE)


# Conversion Tests
def test_ltp_gdc():
    """Test conversion between LTP and Geodetic.

    NOTE: Does not currently test extreme cases where curvatur of
    Earth's surface would have a significant impact. That is generally
    a case for UTM instead of LTP.
    """
    # Times Square
    ltp = pfc.LocalTangentPlane(-73.9864461, 40.7567137)
    gdc = pfc.Geodetic()
    ltp_expected = np.array([
        [196.339820, -302.517581, 15.000],
        [-276.712192,  44.3360115, 30.000],
        [85.2336246,  342.076563, 700.000]
    ])
    gdc_expected = np.array([
        [-73.9841211, 40.7539895, 15.000],
        [-73.9897230, 40.7571129, 30.000],
        [-73.9854367, 40.7597941, 700.000]
    ])

    # Since Geodetic is degrees for x/y, but meters for z, we can't use a single
    # tolerance value sent to `forward_reverse`
    gdc_actual = pfc.CoordinateConverter(ltp, gdc)(ltp_expected)
    np.testing.assert_array_almost_equal(
        gdc_actual[:, :2], gdc_expected[:, :2], __DEGREE_TOLERANCE)  # degree parts
    np.testing.assert_array_almost_equal(
        gdc_actual[:, 2], gdc_expected[:, 2], __METER_TOLERANCE)  # meter part

    ltp_actual = pfc.CoordinateConverter(gdc, ltp)(gdc_expected)
    np.testing.assert_array_almost_equal(
        ltp_actual, ltp_expected, __METER_TOLERANCE)


def test_ltp_gcc():
    """Test conversion between LTP and Geocentric.
    """
    # Times Square
    ltp = pfc.LocalTangentPlane(-73.9864461, 40.7567137)
    gcc = pfc.Geocentric()
    ltp_expected = np.array([
        [196.339, -302.517, 15.000],
        [-276.712,  44.336, 30.000],
        [85.233,  342.076, 700.000]
    ])
    gcc_expected = np.array([
        [1334955.179, -4650677.131,  4141776.030],
        [1334441.149, -4650600.898,  4142048.564],
        [1334875.436, -4650802.007,  4142711.533]
    ])

    forward_reverse(ltp, gcc, ltp_expected, gcc_expected,
                    __METER_TOLERANCE, __METER_TOLERANCE)


def test_ltp_utm():
    """Test conversion between LTP and UTM.
    """
    ltp = pfc.LocalTangentPlane(-73.9864461, 40.7567137)
    utm = pfc.Utm(zone=18, south=False)
    ltp_expected = np.array([
        [196.339, -302.517, 15.000],
        [-276.712,  44.336, 30.000],
        [85.233,  342.076, 700.000]
    ])
    utm_expected = np.array([
        [585754.522, 4511944.464, 15.000],
        [585277.643, 4512285.726, 30.000],
        [585636.015, 4512587.533, 700.000]
    ])
    forward_reverse(ltp, utm, ltp_expected, utm_expected,
                    __METER_TOLERANCE, __METER_TOLERANCE)


def test_gdc_gcc():
    """Test conversion between Geodetic and Geocentric.
    """
    gdc = pfc.Geodetic()
    gcc = pfc.Geocentric()
    gdc_expected = np.array([
        [-80.7746829, 25.5234833, 0.000],
        [-119.9418205, 37.3528545, 0.000],
        [-156.9140529, 70.4281048, 0.000],
        [46.6028192, -19.8187768, 0.000],
        [-70.5889815, -50.4143219, 0.000],
        [-43.7925929, -13.1556067, 0.000]])
    gcc_expected = np.array([
        [923310.410, -5684773.703,  2731518.393],
        [-2533699.562, -4398805.617,  3848595.063],
        [-1971370.755,  -840289.459,  5987207.541],
        [4124143.714,  4361589.576, -2148833.878],
        [1353449.321, -3840969.863, -4892284.831],
        [4484002.834, -4298891.488, -1442173.670]])

    # Since Geodetic is degrees for x/y, but meters for z, we can't use a single
    # tolerance value sent to `forward_reverse`
    gdc_actual = pfc.CoordinateConverter(gcc, gdc)(gcc_expected)
    np.testing.assert_array_almost_equal(
        gdc_actual[:, :2], gdc_expected[:, :2], __DEGREE_TOLERANCE)  # degree parts
    np.testing.assert_array_almost_equal(
        gdc_actual[:, 2], gdc_expected[:, 2], __METER_TOLERANCE)  # meter part

    gcc_actual = pfc.CoordinateConverter(gdc, gcc)(gdc_expected)
    np.testing.assert_array_almost_equal(
        gcc_actual, gcc_actual, __METER_TOLERANCE)


def test_gdc_utm():
    """Test conversion between Geodetic and UTM.
    """
    gdc = pfc.Geodetic()
    utm = pfc.Utm(zone=18, south=False)
    gdc_expected = np.array([
        [-73.9841211, 40.7539895, 15.000],
        [-73.9897230, 40.7571129, 30.000],
        [-73.9854367, 40.7597941, 700.000]
    ])
    utm_expected = np.array([
        [585754.522, 4511944.464, 15.000],
        [585277.643, 4512285.726, 30.000],
        [585636.015, 4512587.533, 700.000]
    ])

    # Since Geodetic is degrees for x/y, but meters for z, we can't use a single
    # tolerance value sent to `forward_reverse`
    gdc_actual = pfc.CoordinateConverter(utm, gdc)(utm_expected)
    np.testing.assert_array_almost_equal(
        gdc_actual[:, :2], gdc_expected[:, :2], __DEGREE_TOLERANCE)  # degree parts
    np.testing.assert_array_almost_equal(
        gdc_actual[:, 2], gdc_expected[:, 2], __METER_TOLERANCE)  # meter part

    utm_actual = pfc.CoordinateConverter(gdc, utm)(gdc_expected)
    np.testing.assert_array_almost_equal(
        utm_actual, utm_actual, __METER_TOLERANCE)


def test_gcc_utm():
    """Test conversion between Geocentric and UTM.
    """
    gcc = pfc.Geocentric()
    utm = pfc.Utm(zone=18, south=False)
    gcc_expected = np.array([
        [1334955.179, -4650677.131,  4141776.030],
        [1334441.149, -4650600.898,  4142048.564],
        [1334875.436, -4650802.007,  4142711.533]
    ])
    utm_expected = np.array([
        [585754.522, 4511944.464, 15.000],
        [585277.643, 4512285.726, 30.000],
        [585636.015, 4512587.533, 700.000]
    ])

    forward_reverse(gcc, utm, gcc_expected, utm_expected,
                    __METER_TOLERANCE, __METER_TOLERANCE)


def test_xyz_in_ltp_to_and_from_gdc():
    """Tests that conversions from LTP to GDC properly handle coordinate order.

    Underlying libraries converting between LTP and GDC expect/provide points in
    (East, North, Up) and (Latitude, Longitude, Altitude), so converters must account
    for the different interpretations.
    """
    one_degree_distances = pfc.compute_degree_size(lat=0)

    convert = pfc.CoordinateConverter(
        pfc.LocalTangentPlane(lon=0, lat=0), pfc.Geodetic())
    eastern_point = convert(
        np.array([[one_degree_distances[0] / 100, 0, 0]]))[0]
    northern_point = convert(
        np.array([[0, one_degree_distances[1] / 100, 0]]))[0]

    assert isclose(eastern_point[0], 0.01, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[1], 0.01, abs_tol=__DEGREE_TOLERANCE)

    convert = pfc.CoordinateConverter(
        pfc.Geodetic(), pfc.LocalTangentPlane(lon=0, lat=0))
    eastern_point = convert(np.array([[0.01, 0, 0]]))[0]
    northern_point = convert(np.array([[0, 0.01, 0]]))[0]

    assert isclose(
        eastern_point[0], one_degree_distances[0] / 100, abs_tol=__METER_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(
        northern_point[1], one_degree_distances[1] / 100, abs_tol=__METER_TOLERANCE)


def test_compute_degree_size():
    """Tests that compute_degree_size is accurate.

    Since there are various methods to calculate the length of a degree
    and constants used in those methods may vary slightly, the
    results are allowed a slightly higher tolerance than __METER_TOLERANCE.
    """
    TOLERANCE = 1
    zero = pfc.compute_degree_size(lat=0)
    fifteen = pfc.compute_degree_size(lat=15)
    thirty = pfc.compute_degree_size(lat=30)
    forty_five = pfc.compute_degree_size(lat=45)
    sixty = pfc.compute_degree_size(lat=60)
    seventy_five = pfc.compute_degree_size(lat=75)

    # Raw values taken from https://www.movable-type.co.uk/scripts/latlong-vincenty.html
    assert isclose(zero[0], 111319.491, abs_tol=TOLERANCE)
    assert isclose(zero[1], 110574.304, abs_tol=TOLERANCE)
    assert isclose(fifteen[0], 107550.397, abs_tol=TOLERANCE)
    assert isclose(fifteen[1], 110648.721, abs_tol=TOLERANCE)
    assert isclose(thirty[0], 96485.974, abs_tol=TOLERANCE)
    assert isclose(thirty[1], 110852.457, abs_tol=TOLERANCE)
    assert isclose(forty_five[0], 78846.335, abs_tol=TOLERANCE)
    assert isclose(forty_five[1], 111131.778, abs_tol=TOLERANCE)
    assert isclose(sixty[0], 55799.47, abs_tol=TOLERANCE)
    assert isclose(sixty[1], 111412.273, abs_tol=TOLERANCE)
    assert isclose(seventy_five[0], 28901.664, abs_tol=TOLERANCE)
    assert isclose(seventy_five[1], 111618.359, abs_tol=TOLERANCE)


def test_dms_to_decimal():
    """Tests that dms_to_decimal performs the correct calculation.
    """
    cases = [
        ((48, 24, 36), 48.41),
        ((48, 24.6), 48.41),
        ((-48, 24, 36), -48.41),
        ((-48, 24.6), -48.41),
        ((30, 36, 4.5), 30.60125),
        ((30, 36.075), 30.60125),
        ((-30, 36, 4.5), -30.60125),
        ((-30, 36.075), -30.60125),
    ]

    for case in cases:
        assert pfc.dms_to_decimal(*case[0]) == case[1]


def test_utm_from_point():
    """Tests that `Utm.from_point` accuractely calculates UTM zone/hemisphere.
    """
    pts = ((-117, 32), (-117, -14), (-73, 38), (-23, -36), (41, 53), (52, -36))
    zones = ((11, False), (11, True), (18, False),
             (27, True), (37, False), (39, True))
    for pt, zone in zip(pts, zones):
        utm = pfc.Utm.from_point(*pt)
        assert utm.zone == zone[0]
        assert utm.south == zone[1]
