"""Utility constants and methods for general use.

Attributes:
    DECIMAL_REGEX (str): A regex that matches decimals.
        These matches may lead with a negative sign, a digit, or a period.
        Scientific notation is supported.
        Ex: 0 1. .2 3.4 5.6e-4
"""
import math
import re
from typing import Tuple

DECIMAL_REGEX = '[+-]?(?:\\.\\d+|\\d+\\.?\\d*)(?:[eE][+-]?\\d+)?'
__EXTENTS_REGEX = re.compile(rf'\(\[({DECIMAL_REGEX}), ?({DECIMAL_REGEX})\], ?\[({DECIMAL_REGEX}), ?({DECIMAL_REGEX})\], ?\[({DECIMAL_REGEX}|nan), ?({DECIMAL_REGEX}|nan)\]\)', re.IGNORECASE)

class ParseError(Exception):
    """Exception for parsing errors.
    """
    def __init__(self, message: str = 'Failed to parse string'):  # pylint: disable=useless-super-delegation
        super().__init__(message)

class Extents:
    """Container for min/max values along three axes.

    Args:
        args: Six numeric values - may be NaN

    Raises:
        RuntimeError: If bad parameters are passed
        TypeError: If any parameters are not numeric
    """
    def __init__(self, *args):
        if len(args) != 6:
            raise RuntimeError('Extents must be insantiated with exactly 6 numbers')

        self.minX, self.maxX, self.minY, self.maxY, self.minZ, self.maxZ = args

        if not all(map(
                lambda x: math.isfinite(x) or math.isinf(x) or math.isnan(x),
                (self.minX, self.maxX, self.minY, self.maxY, self.minZ, self.maxZ))):
            raise TypeError('Cannot instantiate Extents with non-numeric values')

    def __str__(self):
        """Provides string representation of Extents.
        """
        return f'([{self.minX}, {self.maxX}], [{self.minY}, {self.maxY}], [{self.minZ}, {self.maxZ}])'

    def __repr__(self):
        """See ``Extents.__str__``
        """
        return str(self)

    def getMin(self) -> Tuple[float, float, float]:
        """Gets the minimum of the extents.

        Returns:
            Tuple[float,float,float]: The minimum of the extents
        """
        return (self.minX, self.minY, self.minZ)

    def getMax(self) -> Tuple[float, float, float]:
        """Gets the maximum of the extents

        Returns:
            Tuple[float,float,float]: The maximum of the extents
        """
        return (self.maxX, self.maxY, self.maxZ)

    def getCenter(self) -> Tuple[float, float, float]:
        """Gets the center of the extents.

        Returns:
            Tuple[float,float,float]: The center of the extents
        """
        return (self.minX + self.maxX) / 2, (self.minY + self.maxY) / 2, (self.minZ + self.maxZ) / 2

def static_vars(**kwargs):
    """Decorates a function with static variables

    Taken from https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def parseExtents(extents: str) -> Extents:
    """Parses extents from a string.

    Parsed strings are expected to be in the form (with spaces being optional) ([minX, maxX], [minY, maxY], [minZ, maxZ]).
    Z values are optional and may be "NaN" (case insensitive)

    Args:
        extents (str): The extents string to parse

    Returns:
        Extents: The parsed min/max for X, Y, and Z

    Raises:
        ParseError: If the string can't be matched or any values aren't interpretable
    """
    match = __EXTENTS_REGEX.match(extents)
    if not match:
        raise ParseError('Provided extents did not match expected pattern')

    vals = [0] * 6  # Prepare array
    for i in range(1, 7):  # Expect to access match[1] through match[6]
        vals[i-1] = float(match[i])

    return Extents(*vals)
