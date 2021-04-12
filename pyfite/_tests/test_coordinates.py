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
from pyfite.coordinates import computeDegreeSize, CoordinateConverter, CoordinateReferenceSystem, Geocentric, Geodetic, LocalTangentPlane, ProjCrs, Utm

__METER_TOLERANCE = 0.01
__DEGREE_TOLERANCE = 0.000001


__TEST_GDC_POINTS = np.array([
    [-80.77468299836328, 25.523483359092026, 0],
    [-119.94182053217915, 37.35285450844277, 0],
    [-156.91405290143325, 70.42810484231154, 0],
    [24.385561107035954, 69.91032043790399, 0],
    [24.046366314657458, 55.367942013406385, 0],
    [1.9987048100563902, 48.03116705007951, 0],
    [35.918183610676714, 31.626220236696938, 0],
    [113.42419325705953, 23.984269338377466, 0],
    [84.08384371632113, 60.98776445128271, 0],
    [80.86149328421209, 20.690582572441752, 0],
    [-155.4960208145335, 19.54667090833186, 0],
    [117.15533606870929, -29.216448708824693, 0],
    [150.22682832561097, -24.37426106830226, 0],
    [170.23932107594118, -44.45808013822229, 0],
    [20.824015033221407, -32.84849496667563, 0],
    [46.602819253985786, -19.81877685339102, 0],
    [-70.58898151277855, -50.41432195971855, 0],
    [-43.79259291487872, -13.15560679766183, 0]
])


def forward_reverse(crs1: CoordinateReferenceSystem, crs2: CoordinateReferenceSystem, pts1: np.ndarray, pts2: np.ndarray, tol1: float, tol2: float):
    """Tests forward and reverse conversions of points provided two coordinate systems.

    Ensure converting `pts1` via a forward conversion (from `crs1` to `crs2`) yields `pts2` within a tolerance of `tol1`
    Ensure converting `pts2` via a reverse conversion (from `crs2` to `crs1`) yields `pts1` within a tolerance of `tol2`
    """
    forward = CoordinateConverter(crs1, crs2)
    reverse = CoordinateConverter(crs2, crs1)

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
def test_gdc_getProjStr():
    # no offset
    crs = Geodetic()
    assert_proj_str_equivalent(
        crs.getProjStr(), '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    # offset is arbitrary as it does not matter here
    crs = Geodetic(
        offset=(np.random.rand(), np.random.rand(), np.random.rand()))
    assert_proj_str_equivalent(
        crs.getProjStr(), '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


def test_gcc_getProjStr():
    # no offset
    crs = Geocentric()
    assert_proj_str_equivalent(crs.getProjStr(), '+proj=geocent +ellps=WGS84')

    # offset is arbitrary as it does not matter here
    crs = Geocentric(
        offset=(np.random.rand(), np.random.rand(), np.random.rand()))
    assert_proj_str_equivalent(crs.getProjStr(), '+proj=geocent +ellps=WGS84')


def test_utm_getProjStr():
    for zone, south in [(3, False), (12, False), (8, True), (17, True), (43, False), (51, True)]:
        crs_no_off = Utm(zone, south)
        crs_off = Utm(zone, south, offset=(np.random.rand(),
                      np.random.rand(), np.random.rand()))
        utm_str = f'+proj=utm +zone={zone} +ellps=WGS84{" +south" if south else ""}'
        assert_proj_str_equivalent(crs_no_off.getProjStr(), utm_str)
        assert_proj_str_equivalent(crs_off.getProjStr(), utm_str)


def test_proj_getProjStr():
    projs = [
        '+proj=merc +lat_ts=56.5 +ellps=GRS80',
        '+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0',
        '+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
    ]

    for proj in projs:
        assert_proj_str_equivalent(ProjCrs(proj).getProjStr(), proj)
        assert_proj_str_equivalent(
            ProjCrs(proj, offset=(np.random.rand(), np.random.rand(),
                    np.random.rand())).getProjStr(),
            proj)


# Conversion Tests
def test_ltp_gdc():
    # TODO: Include offset cases
    ...


def test_ltp_gcc():
    # TODO: Include offset cases
    ...


def test_ltp_utm():
    # TODO: Include offset cases
    ...


def test_gdc_gcc():
    # TODO: Include offset cases
    from_crs, to_crs = Geodetic(), Geocentric()
    from_pts = np.array([
        [-80.77468299836328, 25.523483359092026, 0],
        [-119.94182053217915, 37.35285450844277, 0],
        [-156.91405290143325, 70.42810484231154, 0],
        [46.602819253985786, -19.81877685339102, 0],
        [-70.58898151277855, -50.41432195971855, 0],
        [-43.79259291487872, -13.15560679766183, 0]])
    to_pts = np.array([
        [923310.41036713, -5684773.70330688,  2731518.39366618],
        [-2533699.56208946, -4398805.61771698,  3848595.06334085],
        [-1971370.75595563,  -840289.45997952,  5987207.54174497],
        [4124143.71477629,  4361589.57680129, -2148833.87802462],
        [1353449.32148221, -3840969.86384632, -4892284.83188213],
        [4484002.83470524, -4298891.48864389, -1442173.67029885]])
    forward_reverse(from_crs, to_crs, from_pts, to_pts, __METER_TOLERANCE, __DEGREE_TOLERANCE)


def test_gdc_utm():
    # TODO: Include offset cases
    ...


def test_gcc_utm():
    # TODO: Include offset cases
    ...


def test_XYZ_in_LocalTangentPlane_to_and_from_Geodetic():
    """Tests that conversions from LTP to GDC properly handle coordinate order.

    Underlying libraries converting between LTP and GDC expect/provide points in
    (East, North, Up) and (Latitude, Longitude, Altitude), so converters must account
    for the different interpretations.
    """
    one_degree_distances = computeDegreeSize(lat=0)

    convert = CoordinateConverter(LocalTangentPlane(lon=0, lat=0), Geodetic())
    eastern_point = convert(
        np.array([[one_degree_distances[0] / 100, 0, 0]]))[0]
    northern_point = convert(
        np.array([[0, one_degree_distances[1] / 100, 0]]))[0]

    assert isclose(eastern_point[0], 0.01, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[1], 0.01, abs_tol=__DEGREE_TOLERANCE)

    convert = CoordinateConverter(Geodetic(), LocalTangentPlane(lon=0, lat=0))
    eastern_point = convert(np.array([[0.01, 0, 0]]))[0]
    northern_point = convert(np.array([[0, 0.01, 0]]))[0]

    assert isclose(
        eastern_point[0], one_degree_distances[0] / 100, abs_tol=__METER_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(
        northern_point[1], one_degree_distances[1] / 100, abs_tol=__METER_TOLERANCE)


def test_computeDegreeSize():
    """Tests that computeDegreeSize is accurate.

    Since there are various methods to calculate the length of a degree
    and constants used in those methods may vary slightly, the
    results are allowed a slightly higher tolerance than __METER_TOLERANCE.
    """
    TOLERANCE = 1
    zero = computeDegreeSize(lat=0)
    fifteen = computeDegreeSize(lat=15)
    thirty = computeDegreeSize(lat=30)
    forty_five = computeDegreeSize(lat=45)
    sixty = computeDegreeSize(lat=60)
    seventy_five = computeDegreeSize(lat=75)

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
