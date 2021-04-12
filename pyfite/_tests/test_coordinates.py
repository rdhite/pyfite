# test_coordinates.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Unit tests for pyfite.coordinates
"""
from cmath import isclose

import numpy as np

import pyfite.coordinates as pfc

__METER_TOLERANCE = 0.1
__DEGREE_TOLERANCE = 0.000001

def test_compute_degree_size():
    """Tests that compute_degree_size is accurate.

    Since there are various methods to calculate the length of a degree
    and constants used in those methods may vary slightly, the
    results are allowed a lightly higher tolerance than __METER_TOLERANCE.
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

def test_XYZ_in_LocalTangentPlane_to_and_from_Geodetic(): # pylint: disable=invalid-name
    """Tests that conversions from LTP to GDC properly handle coordinate order.

    Underlying libraries converting between LTP and GDC expect/provide points in
    (East, North, Up) and (Latitude, Longitude, Altitude), so converters must account
    for the different interpretations.
    """
    one_degree_distances = pfc.compute_degree_size(lat=0)

    convert = pfc.CoordinateConverter(pfc.LocalTangentPlane(lon=0, lat=0), pfc.Geodetic())
    eastern_point = convert(np.array([[one_degree_distances[0] / 100, 0, 0]]))[0]
    northern_point = convert(np.array([[0, one_degree_distances[1] / 100, 0]]))[0]

    assert isclose(eastern_point[0], 0.01, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[1], 0.01, abs_tol=__DEGREE_TOLERANCE)

    convert = pfc.CoordinateConverter(pfc.Geodetic(), pfc.LocalTangentPlane(lon=0, lat=0))
    eastern_point = convert(np.array([[0.01, 0, 0]]))[0]
    northern_point = convert(np.array([[0, 0.01, 0]]))[0]

    assert isclose(eastern_point[0], one_degree_distances[0] / 100, abs_tol=__METER_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[1], one_degree_distances[1] / 100, abs_tol=__METER_TOLERANCE)

def test_projcrs_from_epsg():
    """Tests that a proj string is properly built from EPSG codes.
    """
    epsg4326_crs = pfc.ProjCrs.from_epsg(4326, (60.25, -80.12, 0))
    epsg4978_crs = pfc.ProjCrs.from_epsg(4978, (34.09, -118.13, 0))
    epsg32618_crs = pfc.ProjCrs.from_epsg(32618, (28.54, -81.38, 0))

    assert str(epsg4326_crs) == '+proj=longlat +datum=WGS84 +no_defs +type=crs 60.25 -80.12 0'
    assert str(epsg4978_crs) == '+proj=geocent +datum=WGS84 +units=m +no_defs +type=crs 34.09 -118.13 0'
    assert str(epsg32618_crs) == '+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs +type=crs 28.54 -81.38 0'
    