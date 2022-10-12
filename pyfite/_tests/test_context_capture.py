# test_context_capture.py
# Copyright (c) 2021 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Unit tests for pyfite.context_capture
"""
import os

from pyfite.context_capture import Metadata
from .test_coordinates import assert_proj_str_equivalent

__TESTDATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/testdata/'

def test_context_capture_regex():
    """Tests that the regexes in context_capture work properly.
    """
    metadata_enu = Metadata(__TESTDATA_PATH + 'metadata_enu.xml')
    metadata_enu_offset = Metadata(__TESTDATA_PATH + 'metadata_enu_offset.xml')
    metadata_epsg = Metadata(__TESTDATA_PATH + 'metadata_epsg.xml')
    metadata_epsg_offset = Metadata(__TESTDATA_PATH + 'metadata_epsg_offset.xml')
    metadata_epsg4326 = Metadata(__TESTDATA_PATH + 'metadata_epsg4326.xml')
    metadata_epsg4326_offset = Metadata(__TESTDATA_PATH + 'metadata_epsg4326_offset.xml')

    assert_proj_str_equivalent(str(metadata_enu.get_crs()),
        'ENU -121.4953 34.123 0.0')
    assert_proj_str_equivalent(str(metadata_enu_offset.get_crs()),
        'ENU -100.4124 50.124 0.0 10.0 15.0 0.0')
    assert_proj_str_equivalent(str(metadata_epsg.get_crs()),
        '+proj=utm +zone=30 +south +datum=WGS84 +units=m +no_defs +type=crs')
    assert_proj_str_equivalent(str(metadata_epsg_offset.get_crs()),
        '+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +type=crs 23425.0 623423.0 0.0')
    assert_proj_str_equivalent(str(metadata_epsg4326.get_crs()),
        '+proj=longlat +datum=WGS84 +no_defs +type=crs')
    assert_proj_str_equivalent(str(metadata_epsg4326_offset.get_crs()),
        '+proj=longlat +datum=WGS84 +no_defs +type=crs -91.0 53.0 0.0')
