# test_context_capture.py
# Copyright (c) 2021 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Unit tests for pyfite.context_capture
"""
import os

from pyfite.context_capture import Metadata

__TESTDATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/testdata/'

def test_context_capture_regex():
    metadata_enu = Metadata(__TESTDATA_PATH + 'metadata_enu.xml')
    metadata_enu_offset = Metadata(__TESTDATA_PATH + 'metadata_enu_offset.xml')
    metadata_epsg = Metadata(__TESTDATA_PATH + 'metadata_epsg.xml')
    metadata_epsg_offset = Metadata(__TESTDATA_PATH + 'metadata_epsg_offset.xml')
    metadata_epsg4326 = Metadata(__TESTDATA_PATH + 'metadata_epsg4326.xml')
    metadata_epsg4326_offset = Metadata(__TESTDATA_PATH + 'metadata_epsg4326_offset.xml')

    assert (str(metadata_enu.getCrs()) == 'ENU -121.4953 34.123 0.0')
    assert (str(metadata_enu_offset.getCrs()) == 'ENU -100.4124 50.124 0.0 10.0 15.0 0.0')
    assert (str(metadata_epsg.getCrs()) == '+proj=utm +zone=30 +south +datum=WGS84 +units=m +no_defs +type=crs')
    assert (str(metadata_epsg_offset.getCrs()) == '+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +type=crs 23425.0 623423.0 0.0')
    assert (str(metadata_epsg4326.getCrs()) == '+proj=longlat +datum=WGS84 +no_defs +type=crs')
    assert (str(metadata_epsg4326_offset.getCrs()) == '+proj=longlat +datum=WGS84 +no_defs +type=crs 1423.0 -1205.0 0.0')
