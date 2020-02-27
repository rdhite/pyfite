# pyfite

## Overview
A convenience library for fairly simple, frequently needed operations created for use in expediting work for the Office of Naval Research's (ONR) Future Integrated Training Environment (FITE) contract, hence the name py**fite**. Found to be generally useful, any closed-source or copy(left|right)ed code was removed from the internal repository so it could be provided for use by the public. Typical uses are for coordinate conversions and file copying/moving/locating.

## Examples

### Coordinate Conversions
The following is a (fairly roundabout, just go with it) way to find out the longitude and latitude of Time Square by converting the point (0,0,0) in an offset UTM coordinate space centered at Time Square to a geodetic coordinate space.
```python
import numpy as np
import pyfite as fite
gdc = fite.coordinates.Geodetic() # Standard lat/long coordinate system
utmTimeSquare = fite.coordinates.Utm(zone=18, south=False, offset=(585629, 4512385, 0))
timeSquareToGeodetic = fite.coordinates.CoordinateConverter(utmTimeSquare, gdc)
timeSquareToGeodetic(np.array([[0.0, 0.0, 0.0]])) # array([[-73.98554753, 40.75797057, 0.0]])
```

### File Searching
Presume the following directory layout
```
/some_dir
  bar/
    empty_dir/
  baz/
    moon.jpeg
    star.png
    sun.jpg
  foo/
    foosub/
      im_hiding.jpg
    cool.txt
    some_file_no_ext
  bazfile.txt
  tea_and_crumpets.txt
```
Some searching options look like:
```python
import pyfite as fite
investigator = fite.searching.DirectorySearcher('/some_dir')
investigator.findAll('^baz')
# ['/some_dir/bazfile.txt', '/some_dir/baz/moon.jpeg', '/some_dir/baz/star.png', '/some_dir/baz/stun.jpeg']
investigator.findAll('^baz/') # or investigator.findAll(r'^baz\\') on windows
# ['/some_dir/baz/moon.jpeg', '/some_dir/baz/star.png', '/some_dir/baz/stun.jpeg']
investigator.findAllByExtensions(['jpg', 'jpeg', 'png'])
# ['/some_dir/baz/moon.jpeg', '/some_dir/baz/star.png', '/some_dir/baz/stun.jpeg', '/some_dir/foo/foosub/im_hiding.jpg]
investigator.findFirst('\.txt$')
# '/some_dir/bazfile.txt
investigator.findAll('bar')
# []
```

## License and Copyright
Copyright: Copyright (c) 2020 Applied Research Associates, Inc.

SPDX-License-Identifier: [https://spdx.org/licenses/MIT.html](https://spdx.org/licenses/MIT.html)