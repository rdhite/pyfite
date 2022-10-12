# utils.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

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
__EXTENTS_REGEX = re.compile(rf'\(\[({DECIMAL_REGEX}), ?({DECIMAL_REGEX})\], '
                             rf'?\[({DECIMAL_REGEX}), ?({DECIMAL_REGEX})\], '
                             rf'?\[({DECIMAL_REGEX}|nan), ?({DECIMAL_REGEX}|nan)\]\)',
                             re.IGNORECASE)

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

        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = args

        if not all(map(
                lambda x: math.isfinite(x) or math.isinf(x) or math.isnan(x),
                (self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z))):
            raise TypeError('Cannot instantiate Extents with non-numeric values')

    def __str__(self):
        """Provides string representation of Extents.
        """
        return f'([{self.min_x}, {self.max_x}], [{self.min_y}, {self.max_y}], [{self.min_z}, {self.max_z}])'

    def __repr__(self):
        """See ``Extents.__str__``
        """
        return str(self)

    def get_min(self) -> Tuple[float, float, float]:
        """Gets the minimum of the extents.

        Returns:
            Tuple[float,float,float]: The minimum of the extents
        """
        return (self.min_x, self.min_y, self.min_z)

    def get_max(self) -> Tuple[float, float, float]:
        """Gets the maximum of the extents

        Returns:
            Tuple[float,float,float]: The maximum of the extents
        """
        return (self.max_x, self.max_y, self.max_z)

    def get_center(self) -> Tuple[float, float, float]:
        """Gets the center of the extents.

        Returns:
            Tuple[float,float,float]: The center of the extents
        """
        return (self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2, (self.min_z + self.max_z) / 2

def static_vars(**kwargs):
    """Decorates a function with static variables

    Taken from
        https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    """
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate

def parse_extents(extents: str) -> Extents:
    """Parses extents from a string.

    Parsed strings are expected to be in the form (with spaces being optional):
        ([min_x, max_x], [min_y, max_y], [min_z, max_z])
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
