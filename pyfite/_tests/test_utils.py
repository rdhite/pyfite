"""Unit tests for pyfite.utils
"""
import math
import re

import pytest

import pyfite.utils as pfu

def test_parseExtents():
    """Tests that parseExtents can properly parse.

    Also asserts that the proper exceptions are thrown.
    """
    # First assert proper failure
    assert pytest.raises(pfu.ParseError, pfu.parseExtents, '')                  # Empty String
    assert pytest.raises(pfu.ParseError, pfu.parseExtents, '([1,2],[3,4])')     # Incomplete
    assert pytest.raises(pfu.ParseError, pfu.parseExtents, 'not decimals')

    # Second assert proper parsing
    extents = pfu.parseExtents('([1,2],[3,4],[nan,NaN])')  # Allow NaN for Z axis
    minimum, maximum = extents.getMin(), extents.getMax()
    assert minimum[:2] == (1, 3) and math.isnan(minimum[2])
    assert maximum[:2] == (2, 4) and math.isnan(maximum[2])

    extents = pfu.parseExtents('([3,5],[1,7],[-14,15])')
    assert extents.getMin() == (3, 1, -14)
    assert extents.getMax() == (5, 7, 15)

def test_DECIMAL_REGEX():
    """Tests that all valid C++ versions of decimal strings can be parsed.
    """
    decimal = re.compile(pfu.DECIMAL_REGEX)
    def getValue(s):
        return float(decimal.match(s)[0])

    ones = ['1', '+1', '1.', '+1.', '1.0', '+1.0', '1e0', '+1e0', '1e+0', '1e-0']
    assert all(list(map(lambda x: getValue(x) == 1, ones)))
    assert all(list(map(lambda x: getValue(x.upper()) == 1, ones)))

    tenths = ['0.1', '.1', '+1e-1']
    neg_tenths = ['-0.1', '-.1', '-1e-1']
    assert all(list(map(lambda x: getValue(x) == 0.1, tenths)))
    assert all(list(map(lambda x: getValue(x.upper()) == 0.1, tenths)))
    assert all(list(map(lambda x: getValue(x) == -0.1, neg_tenths)))
    assert all(list(map(lambda x: getValue(x.upper()) == -0.1, neg_tenths)))

    tens = ['10', '+1e+1', '1.0e1', '0.1e+2']
    neg_tens = ['-1.e+1', '-10e0', '-.01e+3']
    assert all(list(map(lambda x: getValue(x) == 10, tens)))
    assert all(list(map(lambda x: getValue(x.upper()) == 10, tens)))
    assert all(list(map(lambda x: getValue(x) == -10, neg_tens)))
    assert all(list(map(lambda x: getValue(x.upper()) == -10, neg_tens)))
