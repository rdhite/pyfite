from pathlib import Path

import numpy as np

from pyfite.obj import Obj

if __name__ == '__main__':
    a = Obj(Path('objs') / 'BUILDING_1776.obj')
    b = Obj(Path('objs') / 'BUILDING_1777.obj')
    c = Obj(Path('objs') / 'BUILDING_1778.obj')
    d = Obj(Path('objs') / 'BUILDING_1779.obj')
    a.combine(b)

    x = Obj(Path('objs') / 'BUILDING_1776.obj')
    y = Obj(Path('objs') / 'BUILDING_1777.obj')
    x.combine_v2(y)
    
    a.write('out/00a.obj')
    x.write('out/00x.obj')
    