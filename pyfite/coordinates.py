# coordinates.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Provides classes for working with varying coordinate reference systems.

Supported types of coordinate reference systems are East-North-Up/Local Tangent Plane,
Geodetic, Geocentric/Earth-Centered Earth-Fixed, UTM, and any valid proj string.

ONLY SUPPORTS WGS84 ELLIPSOID
"""
import math
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pymap3d as p3d
from pyproj import CRS, Transformer

from pyfite.utils import DECIMAL_REGEX, static_vars

__pdoc__ = {}

_OPTIONAL_OFFSET = f'(?: ({DECIMAL_REGEX}) ({DECIMAL_REGEX}) ({DECIMAL_REGEX}))?'

@static_vars(m1=111132.92, m2=-559.82, m3=1.175, m4=-0.0023, p1=111412.84, p2=-93.5, p3=0.118)
def compute_degree_size(lat: float) -> Tuple[float, float]:
    """Computes the size (m) of 1 degree of lon/lat at a given ``lat``.

    Obtained from http://www.csgnetwork.com/degreelenllavcalc.html

    Args:
        lat (float): The latitude at which to compute the size of 1 degree

    Returns:
        Tuple[float,float]: The length of 1 degree longitude and latitude at ``lat``
    """
    lat = math.radians(lat)
    latlen = (compute_degree_size.m1
             + (compute_degree_size.m2 * math.cos(2 * lat))
             + (compute_degree_size.m3 * math.cos(4 * lat))
             + (compute_degree_size.m4 * math.cos(6 * lat)))
    lonlen = ((compute_degree_size.p1 * math.cos(lat))
             + (compute_degree_size.p2 * math.cos(3 * lat))
             + (compute_degree_size.p3 * math.cos(5 * lat)))
    return (lonlen, latlen)

def dms_to_decimal(degrees: int, minutes: Union[int, float] = 0.0, seconds: Optional[float] = 0.0) -> float:
    """Computes decimal degrees from degrees, minutes, and seconds.

    Args:
        degrees (int): The whole portion of DMS
        minutes (Union[int, float]): The minutes portion of DMS.
            May have a fractional portion instead of providing `seconds`
        seconds (float, optional): The seconds portion of DMS

    Returns:
        float: The DMS converted to decimal
    """
    val = abs(degrees) + minutes / 60 + seconds / 3600
    return -val if degrees < 0 else val


class CrsDefError(Exception):
    """Exception for invalid CRS definitions.

    Thrown when a coordinate reference system is unable to parse necessary
    information from a provided string representation.
    """
    def __init__(self, message: str = 'Could not parse provided string representation of coordinate reference system'):  # pylint: disable=useless-super-delegation
        super().__init__(message)


class BadCrsError(Exception):
    """Exception for cases where a CRS is invalid.

    Thrown in cases such as building a converter with only one CoordinateReferenceSystem.
    """
    def __init__(self, message: str = 'A required CRS was not valid for the operation'):  # pylint: disable=useless-super-delegation
        super().__init__(message)


class CoordinateReferenceSystem(ABC):
    """Encompasses information defining a coordinate system.
    """
    @abstractmethod
    def __str__(self):
        """Creates a readable string representing the caller.

        This string will be valid for use in ``CoordinateReferenceSystem.from_str``
        """
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        """Creates a string that evaluates to the caller.
        """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        """Determines if two coordinate systems are equivalent.
        """
        raise NotImplementedError()

    @staticmethod
    def from_str(srep: str) -> 'CoordinateReferenceSystem':
        """Instantiates a ``CoordinateReferenceSystem`` from a string.

        Each of ENU, GCC, GDC, and UTM may all have optional offsets.
        In the examples, GCC, GDC, and UTM have optional offsets shown.

        Args:
            srep (str): A string representation of a ``CoordinateReferenceSystem``

        Examples:
            'ENU -73.985545 40.757978 0'
            'GCC 1334678.23922771, -4650416.284896, 4142132.62185473'
            'GDC -73.985545 40.757978 0'
            'UTM 18N 585554.776, 4512425.71, 0'

        Returns:
            CoordinateReferenceSystem: The interpreted ``CoordinateReferenceSystem``

        Raises:
            CrsDefError: If a ``CoordinateReferenceSystem`` can't be interpreted
        """
        if not srep:
            raise CrsDefError('Cannot instantiate a CoordinateReferenceSystem with an empty string representation')

        crs = None
        if re.match('^(?:ltp|enu)', srep, re.IGNORECASE):
            crs = LocalTangentPlane.from_str(srep)
        elif re.match('^(?:geodetic|gdc|lla)', srep, re.IGNORECASE):
            crs = Geodetic.from_str(srep)
        elif re.match('^(?:utm)', srep, re.IGNORECASE):
            crs = Utm.from_str(srep)
        elif re.match('^(?:ecef|gcc)', srep, re.IGNORECASE):
            crs = Geocentric.from_str(srep)
        else:
            raise CrsDefError('Unknown CoordinateReferenceSystem definition string')

        return crs

    @staticmethod
    def find_str(string: str) -> str:
        """Attempts to locate a pyfite compatible CRS substring within ``string``
        """
        # NOTE: _srepRegex is always compiled without case sensitivity, so we must recompile
        # because Pattern[str].pattern does NOT indicate case sensitivities
        pattern = re.compile(f'(?:{LocalTangentPlane._srepRegex.pattern})|'   # pylint: disable=protected-access
                             f'(?:{Geocentric._srepRegex.pattern})|'          # pylint: disable=protected-access
                             f'(?:{Geodetic._srepRegex.pattern})|'            # pylint: disable=protected-access
                             f'(?:{Utm._srepRegex.pattern})', re.IGNORECASE)  # pylint: disable=protected-access
        m = re.search(pattern, string)
        return m[0] if m else ''

    @abstractmethod
    def get_proj_str(self):
        """Returns the proj string representation of the caller.

        Warn:
            Proj does not support arbitrary local offsets, so the returned
            string **will not** account for any offset of the caller.

        Returns:
            str: The proj string representation of the caller

        Raises:
            RuntimeError: If the caller doesn't have a valid proj representation.
                          Likely causes are having a local offset or being a LocalTangentPlane.
        """
        raise RuntimeError(f'Class {type(self).__name__} ')

    @property
    def offset(self) -> Tuple[float, float, float]:
        """Offset to be assumed relative to a global coordinate system.

        Offsets aren't applicable to all sub-classes (e.g. `ProjCrs`), so an offset member is
        not defined by default and the setter must be used prior to getting. This should be
        done in sub-class __init__ methods.

        Raises:
            AttributeError
        """
        try:
            return self._offset
        except AttributeError as e:
            raise AttributeError(f'Instance of {type(self).__name__} ') from e

    @offset.setter
    def offset(self, offset: Tuple[float, float, float]):
        self._offset = offset

    def _has_offset(self) -> bool:
        """Determines if the caller has an offset.
        """
        return self.offset != (0.0, 0.0, 0.0)

    def _get_offset_str(self):
        """Creates the offset string for __str__.
        """
        s = ''
        if self._has_offset():
            s = ' {} {} {}'.format(*self.offset)
        return s

    def _get_offset_repr(self, include_comma: Optional[bool] = True):
        """Creates the offset string for __repr__.
        """
        s = ''
        if self._has_offset():
            s = (',' if include_comma else '') + f'offset={self.offset}'
        return s


class LocalTangentPlane(CoordinateReferenceSystem):
    """A cartesian coordinate system tangent to the ellipsoid surface.

    Represents a cartesian coordinate system where the x- and y-
    axes define a plane tangent to the surface of an ellipsoid with
    the x-axis representing East(+) and West(-), the y-axis
    representing North(+) and South(-), and the z-axis representing
    altitude.

    Args:
        lon (float): The longitude of the tangent plane origin
        lat (float): The latitude of the tangent plane origin
        alt (float, optional): The altitude above the WGS84 ellipsoid of the tangent plane origin
        offset (Tuple[float,float,float], optional): The offset by which points are adjusted
    """
    _srepRegex = re.compile(f'(?:ltp|enu) ({DECIMAL_REGEX}) ({DECIMAL_REGEX}) ({DECIMAL_REGEX}){_OPTIONAL_OFFSET}',
                            re.IGNORECASE)

    def __init__(self, lon: float, lat: float, alt: Optional[float] = 0.0,
                 offset: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self.lon, self.lat, self.alt, self._offset = lon, lat, alt, offset

    def __str__(self):
        """See ``CoordinateReferenceSystem.__str__``.
        """
        return f'ENU {self.lon} {self.lat} {self.alt}' + self._get_offset_str()

    def __repr__(self):
        """See ``CoordinateReferenceSystem.__repr__``.
        """
        return f'LocalTangentPlane(lon={self.lon},lat={self.lat},alt={self.alt}' + self._get_offset_repr() + ')'

    def __eq__(self, other: CoordinateReferenceSystem) -> bool:
        """See ``CoordinateReferenceSystem.__eq__``.
        """
        return isinstance(other, LocalTangentPlane) and self.offset == other.offset

    @staticmethod
    def from_str(srep: str) -> 'LocalTangentPlane':
        """See ``CoordinateReferenceSystem.from_str``.
        """
        sm = LocalTangentPlane._srepRegex.match(srep)
        if not sm:
            raise CrsDefError(f'Could not parse provided string representation for {LocalTangentPlane}: {srep}')

        lon, lat, alt, offset = float(sm[1]), float(sm[2]), float(sm[3]), (0.0, 0.0, 0.0)
        if sm[4]:
            # The regex is defined such that either all 3 offset parameters are available, or none are
            offset = (float(sm[4]), float(sm[5]), float(sm[6]))
        return LocalTangentPlane(lon, lat, alt, offset)

    def get_proj_str(self):
        """Raise exception due to lack of proj support.

        Raises:
            RuntimeError: Because proj does not support local tangent planes
        """
        raise RuntimeError('Local tangent planes are not directly supported by Proj')


class Geocentric(CoordinateReferenceSystem):
    """The Earth-Centered Earth-Fixed coordinate system.

    Represents a coordinate system centered at the center of Earth. The x-axis
    intersects the Equator and the Prime Meridian, the y-axis intersects the
    Equator and +90 longitude, and the z-axis intersects the North Pole.

    Args:
        offset (Tuple[float, float, float], optional): The offset by which points are adjusted
    """
    _srepRegex = re.compile(f'(?:gcc|geocentric|ecef){_OPTIONAL_OFFSET}', re.IGNORECASE)

    def __init__(self, offset: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self._offset = offset

    def __str__(self):
        """See ``CoordinateReferenceSystem.__str__``.
        """
        return 'GCC' + self._get_offset_str()

    def __repr__(self):
        """See ``CoordinateReferenceSystem.__repr__``.
        """
        return 'Geocentric(' + self._get_offset_repr(include_comma=False) + ')'

    def __eq__(self, other: CoordinateReferenceSystem) -> bool:
        """See ``CoordinateReferenceSystem.__eq__``.
        """
        return isinstance(other, Geocentric) and self.offset == other.offset

    @staticmethod
    def from_str(srep: str) -> 'Geocentric':
        """See ``CoordinateReferenceSystem.from_str``.
        """
        sm = Geocentric._srepRegex.match(srep)
        if not sm:
            raise CrsDefError(f'Could not parse provided string representation for {Geocentric}: {srep}')

        offset = (0.0, 0.0, 0.0)
        if sm[1]:
            offset = (float(sm[1]), float(sm[2]), float(sm[3]))
        return Geocentric(offset)

    def get_proj_str(self) -> str:
        """See ``CoordinateReferenceSystem.get_proj_str``.
        """
        return '+proj=geocent +ellps=WGS84'


class Geodetic(CoordinateReferenceSystem):
    """The longitude, latitude, altitude coordinate system.

    Args:
        offset (Tuple[float, float, float], optional): The offset by which points are adjusted
    """
    _srepRegex = re.compile(f'(?:gdc|geodetic|lla){_OPTIONAL_OFFSET}', re.IGNORECASE)

    def __init__(self, offset: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self._offset = offset

    def __str__(self):
        """See ``CoordinateReferenceSystem.__str__``.
        """
        return 'GDC' + self._get_offset_str()

    def __repr__(self):
        """See ``CoordinateReferenceSystem.__repr__``.
        """
        return 'Geodetic(' + self._get_offset_repr(include_comma=False) + ')'

    def __eq__(self, other: CoordinateReferenceSystem) -> bool:
        """See ``CoordinateReferenceSystem.__eq__``.
        """
        return isinstance(other, Geodetic) and self.offset == other.offset

    @staticmethod
    def from_str(srep: str) -> 'Geodetic':
        """See ``CoordinateReferenceSystem.from_str``.
        """
        sm = Geodetic._srepRegex.match(srep)
        if not sm:
            raise CrsDefError(f'Could not parse provided string representation for {Geodetic}: {srep}')

        offset = (0.0, 0.0, 0.0)
        if sm[1]:
            offset = (float(sm[1]), float(sm[2]), float(sm[3]))
        return Geodetic(offset)

    def get_proj_str(self) -> str:
        """See ``CoordinateReferenceSystem.get_proj_str``.
        """
        return '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'


class Utm(CoordinateReferenceSystem):
    """The Universal Trasverse Mercator coordinate system.

    Args:
        zone (int)
        south (bool): Whether to use the south half of ``zone`` or not
        offset (Tuple[float,float,float], optional): The offset by which points are adjusted
    """
    _srepRegex = re.compile(rf'(?:utm) (\d{{1,2}})([A-z]){_OPTIONAL_OFFSET}', re.IGNORECASE)

    def __init__(self, zone: int, south: bool, offset: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self.zone, self.south, self._offset = zone, south, offset

    def __str__(self):
        """See ``CoordinateReferenceSystem.__str__``.
        """
        return 'UTM {}{}'.format(self.zone, 'S' if self.south else 'N') + self._get_offset_str()

    def __repr__(self):
        """See ``CoordinateReferenceSystem.__repr__``.
        """
        return f'Utm(zone={self.zone},south={self.south}' + self._get_offset_repr() + ')'

    def __eq__(self, other: CoordinateReferenceSystem) -> bool:
        """See ``CoordinateReferenceSystem.__eq__``.
        """
        return isinstance(other, Utm) and self.offset == other.offset

    @staticmethod
    def from_str(srep: str) -> 'Utm':
        """See ``CoordinateReferenceSystem.from_str``.
        """
        sm = Utm._srepRegex.match(srep)
        if not sm:
            raise CrsDefError(f'Could not parse provided string representation for {Utm}: {srep}')

        offset = (0.0, 0.0, 0.0)
        if sm[3]:
            offset = (float(sm[3]), float(sm[4]), float(sm[5]))
        return Utm(int(sm[1]), sm[2].upper() <= 'M', offset)

    @staticmethod
    def from_point(lon: float, lat: float, _=None) -> 'Utm':
        """Construct a ``Utm`` from a given ``lon`` and ``lat``.

        NOTE: Proj does not take into account the special case for 31V/32V
        in the EU, so this method does not either.
        """
        # Note that a third argument is provided as optional for cases in which
        # a 3d point is expanded with *

        if not -90 <= lat <= 90:
            raise ValueError('Latitude must be within the range -90 to 90')

        lon = lon % 360 # Bring into 0-360
        lon = lon - 360 if lon > 180 else lon # Bring into -180 - 180
        return Utm(math.floor((lon + 180) / 360 * 60) + 1, lat < 0)

    def get_proj_str(self) -> str:
        """See ``CoordinateReferenceSystem.get_proj_str``
        """
        proj_str = f'+proj=utm +zone={self.zone} +ellps=WGS84'
        if self.south:
            proj_str += ' +south'
        return proj_str


class ProjCrs(CoordinateReferenceSystem):
    """Any arbitrary coordinate system supported by Proj.

    NOTE: Offsets can be provided but are interpreted and applied completely outside
    of any proj-based context.


    Args:
        proj_str (str): A valid Proj string
        offset (Tuple[float, float, float], optional): The offset by which points are adjusted
    """
    def __init__(self, proj_str: str, offset: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self._proj = proj_str
        self._offset = offset

    def __str__(self):
        """See ``CoordinateReferenceSystem.__str__``.
        """
        return self._proj + self._get_offset_str()

    def __repr__(self):
        """See ``CoordinateReferenceSystem.__repr__``.
        """
        return f'ProjCrs({self._proj})'

    def __eq__(self, other):
        """See ``CoordinateReferenceSystem.__eq__``.
        """
        return isinstance(other, ProjCrs) and self._proj == other._proj and self._offset == other._offset  # pylint: disable=protected-access

    @staticmethod
    def from_str(srep: str) -> 'ProjCrs':
        """See ``CoordinateReferenceSystem.from_str``
        """
        return ProjCrs(srep)

    @staticmethod
    def from_epsg(code: Union[str,int], offset: Tuple[float, float, float]) -> 'ProjCrs':
        """Creates ProjCrs from an EPSG code.

        Args:
            code: The EPSG code to use
        
        Returns:
            A ProjCrs object representing the `code`
        """
        return ProjCrs(CRS.from_epsg(code).to_proj4(), offset)

    def get_proj_str(self) -> str:
        """See ``CoordinateReferenceSystem.get_proj_str``
        """
        return self._proj


class CoordinateConverter:
    """Converts points from one CRS to another.

    This class is callable and, when called with points,
    converts points from the ``from_crs`` to the ``to_crs``.
    This class may also use the ``convert`` method to achieve
    the same result.

    Args:
        from_crs: The CRS to convert points from
        to_crs: The CRS to convert points to
    """

    def __init__(self, from_crs: Union[CoordinateReferenceSystem, str], to_crs: Union[CoordinateReferenceSystem, str]):
        from_crs = (from_crs if isinstance(from_crs, CoordinateReferenceSystem)
                           else CoordinateReferenceSystem.from_str(from_crs))
        to_crs = to_crs if isinstance(to_crs, CoordinateReferenceSystem) else CoordinateReferenceSystem.from_str(to_crs)

        self.__from_offset = from_crs.offset
        self.__to_offset = to_crs.offset

        self.__convert = CoordinateConverter.__get_converter(from_crs, to_crs)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Convert a set of points to the targeted CRS.

        Args:
            points: A numpy array of 3D points with shape (N, 3)

        Returns:
            A set of points converted to the target CRS with shape (N, 3)
        """
        shape = np.asarray(points).shape
        if len(shape) != 2 or shape[1] != 3:
            raise RuntimeError(f'Cannot convert non-3D points: shape was {shape}')

        # Conversions are almost always done using pyproj, which can't handle arbitrary offsets
        # on the three axes very well. Instead, do the offset math outside of the conversion
        # so that pyproj can operate in its normal way
        return self.__convert(points + self.__from_offset) - self.__to_offset

    def convert(self, points: np.ndarray) -> np.ndarray:
        """Convert points according to coordinate systems of construction.

        Args:
            points: A numpy array of 3D points with shape (N, 3)

        Returns:
            A set of points converted to the target CRS with shape (N, 3)
        """
        return self.__call__(points)

    @staticmethod
    def __get_converter(from_crs: CoordinateReferenceSystem,
            to_crs: CoordinateReferenceSystem) -> Callable[[np.ndarray], np.ndarray]:
        """Gets a function that converts points from ``from_crs`` to ``to_crs``.

        Args:
            from_crs (CoordinateReferenceSystem): The source coordinate system
            to_crs (CoordinateReferenceSystem): The target coordinate system

        Returns:
            Callable[[np.ndarray],np.ndarray]: Method that converts a set of points
        """
        # note that offsets are handled in __call__ so they aren't here
        func = None
        if (not isinstance(from_crs, LocalTangentPlane)) and (not isinstance(to_crs, LocalTangentPlane)):
            func = CoordinateConverter.__get_pyproj_func(from_crs.get_proj_str(), to_crs.get_proj_str())

        # Anything dealing with a local tangent plane isn't supported by PROJ, so use pymap3d
        elif isinstance(from_crs, LocalTangentPlane):

            if isinstance(to_crs, LocalTangentPlane):

                def conv_func(points):
                    x, y, z = p3d.enu2ecef(
                        points[:, 0], points[:, 1], points[:, 2],
                        from_crs.lat, from_crs.lon, from_crs.alt)
                    x, y, z = p3d.ecef2enu(x, y, z, to_crs.lat, to_crs.lon, to_crs.alt)
                    return np.column_stack((x, y, z))
                func = conv_func

            elif isinstance(to_crs, Geocentric):

                def conv_func(points):
                    x, y, z = p3d.enu2ecef(points[:, 0], points[:, 1], points[:, 2],
                                           from_crs.lat, from_crs.lon, from_crs.alt)
                    return np.column_stack((x, y, z))
                func = conv_func

            elif isinstance(to_crs, Geodetic):

                def conv_func(points):
                    lat, lon, alt = p3d.enu2geodetic(points[:, 0], points[:, 1], points[:, 2],
                                                     from_crs.lat, from_crs.lon, from_crs.alt)
                    return np.column_stack((lon, lat, alt))
                func = conv_func

            elif isinstance(to_crs, Utm):

                def conv_func(points):
                    lat, lon, alt = p3d.enu2geodetic(points[:, 0], points[:, 1], points[:, 2],
                                                     from_crs.lat, from_crs.lon, from_crs.alt)
                    return CoordinateConverter.__get_pyproj_func(
                        Geodetic().get_proj_str(),
                        to_crs.get_proj_str())(np.column_stack((lon, lat, alt)))
                func = conv_func

        elif isinstance(from_crs, Geocentric):
            def conv_func(points):
                x, y, z = p3d.ecef2enu(points[:, 0], points[:, 1], points[:, 2], to_crs.lat, to_crs.lon, to_crs.alt)
                return np.column_stack((x, y, z))
            func = conv_func

        elif isinstance(from_crs, Geodetic):
            def conv_func(points):
                e, n, u = p3d.geodetic2enu(points[:, 1], points[:, 0], points[:, 2], to_crs.lat, to_crs.lon, to_crs.alt)
                return np.column_stack((e, n, u))
            func = conv_func

        elif isinstance(from_crs, Utm):
            def conv_func(points):
                points = CoordinateConverter.__get_pyproj_func(
                    from_crs.get_proj_str(),
                    Geodetic().get_proj_str())(points)

                # geodetic2enu goes lat/long instead of long/lat, so flip axes 0 and 1
                e, n, u = p3d.geodetic2enu(points[:, 1], points[:, 0], points[:, 2], to_crs.lat, to_crs.lon, to_crs.alt)

                return np.column_stack((e, n, u))
            func = conv_func

        if not func:
            raise RuntimeError(f'Could not determine a conversion function from {from_crs} to {to_crs}')

        return func

    @staticmethod
    def __get_pyproj_func(from_crs: str, to_crs: str) -> Callable[[np.ndarray], np.ndarray]:
        """Creates a function that converts using Proj strings.

        Args:
            from_crs (str): Proj string of the source coordinate system
            to_crs (str): Proj string of the target coordinate system

        Returns:
            Callable[[np.ndarray],np.ndarray]: A method that converts a set of points
        """
        def conv_func(points):
            transformer = Transformer.from_crs(CRS.from_string(from_crs), CRS.from_string(to_crs), always_xy=True)
            return np.column_stack(transformer.transform(points[:, 0], points[:, 1], points[:, 2]))
        return conv_func
__pdoc__['CoordinateConverter.__call__'] = CoordinateConverter.__call__.__doc__
