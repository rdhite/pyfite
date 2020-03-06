# context_capture.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Classes and methods for interacting with Context Capture products.
"""
import re
from pathlib import Path
from typing import Tuple, Union

from pyfite.coordinates import CoordinateReferenceSystem, Geodetic, LocalTangentPlane
from pyfite.utils import DECIMAL_REGEX

class Metadata:
    """A convenience class for interacting with metadata.xml files generated by ContextCapture.

    Note:
        This class does not guarantee tracking of all information in a metadata.xml.
        It is only guaranteed to track the <SRS> and <SRSOrigin> tags and enough of the DOM
        to be able to recreate a minimal metadata.xml should those tags be altered.

    Args:
        path (Union[str, Path]): The path to the metadata.xml to parse
    """
    __enuRegex = f'<SRS>ENU:({DECIMAL_REGEX}),({DECIMAL_REGEX})</SRS>'
    __geodeticRegex = '<SRS>EPSG:4326</SRS>'
    __offsetRegex = f'<SRSOrigin>({DECIMAL_REGEX}),({DECIMAL_REGEX}),({DECIMAL_REGEX})</SRSOrigin>'

    def __init__(self, path: Union[str, Path]):
        self._crs = None

        with open(path, 'r') as f:
            metadata = f.read()

            # Find the offset, which is present for all coordinate systems
            offset = (0.0, 0.0, 0.0)
            m = re.search(Metadata.__offsetRegex, metadata)
            if m:
                offset = (float(m[1]), float(m[2]), float(m[3]))

            # First check to see if this is an ENU metadata.xml
            m = re.search(Metadata.__enuRegex, metadata)
            if m:
                lat, lon = float(m[1]), float(m[2])
                self._crs = LocalTangentPlane(lon, lat, 0.0, offset)
                return

            # Next check to see if this is a geodetic metadata.xml
            m = re.search(Metadata.__geodeticRegex, metadata)
            if m:
                self._crs = Geodetic(offset)

    def getOffset(self) -> Tuple[float, float, float]:
        """Gets the offset specified by the metadata.xml.

        Returns:
            Tuple[float,float,float]: The offset specified in the metadata.xml
        """
        return self._crs.offset

    def setOffset(self, offset: Tuple[float, float, float]) -> None:
        """Sets the offset.

        Args:
            offset (Tuple[float,float,float]): The offset the metadata should now assume
        """
        self._crs.offset = offset

    def getCrs(self) -> CoordinateReferenceSystem:
        """Gets the CoordinateReferenceSystem represented by a Metadata.

        Returns:
            CoordinateReferenceSystem: The CoordinateReferenceSystem parsed from the metadata.xml
        """
        return self._crs

    def setCrs(self, crs: CoordinateReferenceSystem) -> None:
        """Sets the CoordinateReferenceSystem.

        Args:
            crs (CoordinateReferenceSystem): The CoordinateReferenceSystem the metadata should now assume
        """
        self._crs = crs

    def write(self, path: Union[str, Path]) -> None:
        """Writes the metadata out to ``path``.

        Warn:
            This method may lost information present in the original metadata.xml. It is only guaranteed
            to track Coordinate Reference System information.

        Note: It is assumed that all parents of ``path`` already exist.

        Args:
            path (Union[str,Path]): The path to write to
        """
        raise RuntimeError('Metadata.write not implemented')
