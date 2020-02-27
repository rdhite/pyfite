"""Unit tests for pyfite.coordinates
"""
from cmath import isclose

import numpy as np
from pyfite.coordinates import computeDegreeSize, CoordinateConverter, Geocentric, Geodetic, LocalTangentPlane, Utm

__METER_TOLERANCE = 0.1
__DEGREE_TOLERANCE = 0.000001

def test_computeDegreeSize():
    """Tests that computeDegreeSize is accurate.

    Since there are various methods to calculate the length of a degree
    and constants used in those methods may vary slightly, the
    results are allowed a lightly higher tolerance than __METER_TOLERANCE.
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

def test_XYZ_in_LocalTangentPlane_to_and_from_Geodetic():
    """Tests that conversions from LTP to GDC properly handle coordinate order.

    Underlying libraries converting between LTP and GDC expect/provide points in
    (East, North, Up) and (Latitude, Longitude, Altitude), so converters must account
    for the different interpretations.
    """
    one_degree_distances = computeDegreeSize(lat=0)

    convert = CoordinateConverter(LocalTangentPlane(lon=0, lat=0), Geodetic())
    eastern_point = convert(np.array([[one_degree_distances[0] / 100, 0, 0]]))[0]
    northern_point = convert(np.array([[0, one_degree_distances[1] / 100, 0]]))[0]

    assert isclose(eastern_point[0], 0.01, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__DEGREE_TOLERANCE)
    assert isclose(northern_point[1], 0.01, abs_tol=__DEGREE_TOLERANCE)

    convert = CoordinateConverter(Geodetic(), LocalTangentPlane(lon=0, lat=0))
    eastern_point = convert(np.array([[0.01, 0, 0]]))[0]
    northern_point = convert(np.array([[0, 0.01, 0]]))[0]

    assert isclose(eastern_point[0], one_degree_distances[0] / 100, abs_tol=__METER_TOLERANCE)
    assert isclose(eastern_point[1], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[0], 0, abs_tol=__METER_TOLERANCE)
    assert isclose(northern_point[1], one_degree_distances[1] / 100, abs_tol=__METER_TOLERANCE)
