# obj.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Contains classes for reading, manipulating, and writing .obj files.
"""
import os
import shutil
from collections import namedtuple
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from pyfite.coordinates import CoordinateConverter, CoordinateReferenceSystem
from pyfite.utils import Extents


class Obj:  # pylint: disable=too-many-instance-attributes
    """Supports reading and writing of .obj files

    Material definitions and texture files are tracked and copied to output
    directories on writes. Additionally, if a base CoordinateReferenceSystem
    is provided, Objs can be converted to other CoordinateReferenceSystems.

    Attributes:
        INV_IDX (np.uint32): Value representing an invalid index
        vertices (numpy.array): Array of vertices
        tex_coords (numpy.array): Array of texture coordinates
        faces (numpy.array): Array of face
            [
                [vIdx, vtIdx, vnIdx],  # first vertex of face
                [vIdx, vtIdx, vnIdx],  # second vertex of face
                [vIdx, vtIdx, vnIdx]   # third vertex of face
            ]
        materials (List[List[Obj.MtlLib,str,int]]): Record of materials. Each entry is:
            [
                MtlLib instance,  # See ``MtlLib`` below
                material name,    # The name of the material in the .mtl
                starting index    # The index of the first face using this material
            ]

    Types:
        MtlLib: namedtuple for tracking material libraries
            Attributes:
                base (Path): the base directory containing the .obj referencing the .mtl
                relative (Union[str,Path]): the relative path to the .mtl from ``base``



    Args:
        path (Union[Path,str], optional): The path to an .obj to read
        crs (CoordinateReferenceSystem, optional): The underlying CRS of the Obj

    Raises:
        FileNotFoundError: The ``path``, if specified, does not exist
    """
    INV_IDX = np.uint32(-1)

    MtlLib = namedtuple('MtlLib', ['base', 'relative'])  # [Path, Union[str,Path]]

    def __init__(self, path: Optional[Union[str, Path]] = None, crs: CoordinateReferenceSystem = None):
        self._crs = crs
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.tex_coords = np.empty((0, 2), dtype=np.float32)
        self.normals = np.empty((0, 3), dtype=np.float32)
        self.faces = np.empty((0, 3, 3), dtype=np.uint32)
        self.materials: List[List[Obj.MtlLib, str, int]] = []  # [mtllib, material, ending idx]
        self.__root: Path = None
        self.__have_tex, self.__have_norm = False, False

        if path:
            if os.path.isfile(path):
                self.read(path)
            else:
                raise FileNotFoundError(f'{path} does not exist')

    def __process_lines(self, content: List[str]) -> None:
        """Processes each line from an obj.

        Args:
            content (List[str]): The lines of an obj
        """
        attributes = {'v': 0, 'vt': 0, 'vn': 0, 'f': 0, 'mtl': '', 'mtllib': ''}
        for line in content:
            self._custom_line_processing(line)
            tokens = line.split()

            if len(tokens) == 0 or tokens[0][0] == '#':
                continue

            attribute = tokens[0]

            if attribute == 'v':
                self.vertices[attributes['v']] = (np.float32(tokens[1]), np.float32(tokens[2]), np.float32(tokens[3]))
                attributes['v'] += 1

            elif attribute == 'vt':
                self.tex_coords[attributes['vt']] = (np.float32(tokens[1]), np.float32(tokens[2]))
                attributes['vt'] += 1

            elif attribute == 'vn':
                self.normals[attributes['vn']] = (np.float32(tokens[1]), np.float32(tokens[2]), np.float32(tokens[3]))
                attributes['vn'] += 1

            elif attribute == 'mtllib':
                attributes['mtllib'] = tokens[1]
                self.__record_mat_change(attributes['mtllib'], attributes['mtl'], attributes['f'])

            elif attribute == 'usemtl':
                attributes['mtl'] = tokens[1]
                self.__record_mat_change(attributes['mtllib'], attributes['mtl'], attributes['f'])

            elif attribute == 'f':
                for i, token in enumerate(tokens[1:]):
                    vf = [Obj.INV_IDX] * 3
                    for j, val in enumerate(token.split('/')):
                        if val:
                            vf[j] = int(val)
                    self.faces[attributes['f']][i] = vf
                attributes['f'] += 1

    def _custom_line_processing(self, line: str) -> None:
        """Performs any custom logic for a line while reading.

        Args:
            line (str): The line to process
        """
        pass  # pylint: disable=unnecessary-pass

    def _determine_if_have_tex_norm(self) -> None:
        """Updates ``__have_tex`` and ``__have_norm`` to their appropriate values.
        """
        [_, self.__have_tex, self.__have_norm] = np.any(np.apply_along_axis(
            lambda x: any(y != Obj.INV_IDX for y in x), 0, self.faces), 0)

    def read(self, path: Union[str, Path]) -> None:
        """Reads an OBJ into memory.

        Note:
            Only supports 'v', 'vt', 'vn', 'f', 'mtllib', and 'usemtl'

        Warn:
            Vertex normals are not currently supported during coordinate conversions

        Args:
            path (Union[str,Path]): The path to the obj to read
        """
        if isinstance(path, str):
            path = Path(path)
        self.__root = path.parent

        # First get a count of everything we need to store
        f = open(path, 'r')
        content = f.readlines()
        f.close()

        counts = {'v': 0, 'vt': 0, 'vn': 0, 'f': 0, 'usemtl': 0, 'mtllib': 0}
        for line in content:
            key = line[:line.find(' ')]
            if key in counts:
                counts[key] += 1

        self.vertices = np.empty((counts['v'], 3), dtype=np.float32)
        self.tex_coords = np.empty((counts['vt'], 2), dtype=np.float32)
        self.normals = np.empty((counts['vn'], 3), dtype=np.float32)
        self.faces = np.empty((counts['f'], 3, 3), dtype=np.uint32)

        self.__process_lines(content)

        # Clear out an empty material if it happened to happen at the end
        if (len(self.materials) >= 2 and self.materials[-1][2] == self.materials[-2][2] or
                len(self.materials) > 0 and self.materials[-1][2] == len(self.faces)):
            self.materials.pop()

        self._remove_duplicate_materials()

        self._determine_if_have_tex_norm()

    def _remove_duplicate_materials(self) -> None:
        """Group faces by materials used and remove any duplicate materials
        """
        all_mtl_names = [n for _, n, _ in self.materials]
        uniq = list(set(all_mtl_names))

        if len(uniq) == len(all_mtl_names):
            return

        face_group = {}
        for n in uniq:
            face_group[n] = np.empty((0, 3, 3), dtype=np.uint32)

        for mtl_i in range(len(all_mtl_names)):
            mtl_face_i = self.materials[mtl_i][2]
            mtl_face_j = self.materials[mtl_i + 1][2] if mtl_i + 1 < len(self.materials) else len(self.faces)
            face_group[all_mtl_names[mtl_i]] = np.concatenate((face_group[all_mtl_names[mtl_i]], self.faces[mtl_face_i:mtl_face_j]))

        new_materials = []
        starting_i = 0
        for mtl_name in uniq:
            new_materials.append((self.materials[all_mtl_names.index(mtl_name)][0], mtl_name, starting_i))
            starting_i = starting_i + len(face_group[mtl_name])

        new_faces = np.empty((0, 3, 3), dtype=np.uint32)
        for arr in face_group.values():
            new_faces = np.concatenate((new_faces, arr))

        self.faces = new_faces
        self.materials = new_materials


    def write(self, dest: Union[str, Path, TextIOWrapper], precision: Union[int, Tuple[int, int, int]] = None) -> None:
        """Writes the Obj and copies textures with it.

        Note:
            Order of data may not be identical to any OBJs that were read.
            All 'v', then 'vt', then 'vn', then 'f/mtllib/usemtl' are written.

        Args:
            dest (Union[str,Path,TextIOWrapper]): The destination to write to
            precision (Union[int,Tuple[int,int,int]], optional): The number of decimal
                places to write for v, vt, and vn respectively
        """
        if not isinstance(precision, tuple):
            precision = (precision, precision, precision)
        if not all(map(lambda x: isinstance(x, (int, type(None))), precision)):
            raise ValueError(f'Provided precision was not an int or None: {precision}')

        self._determine_if_have_tex_norm()

        if isinstance(dest, str):
            dest = Path(dest)

        if isinstance(dest, TextIOWrapper):
            self._write_v(dest, precision[0])
            self._write_vt(dest, precision[1])
            self._write_vn(dest, precision[2])
            self._write_f(dest)
            dest_parent = Path(dest.name).parent
        else:
            dest_parent = dest.parent
            os.makedirs(dest.parent, exist_ok=True)
            with open(dest, 'w+', 8192) as obj:
                self._write_v(obj, precision[0])
                self._write_vt(obj, precision[1])
                self._write_vn(obj, precision[2])
                self._write_f(obj)

        self._copy_materials(dest_parent)

    def _write_v(self, fout: TextIOWrapper, precision: int) -> None:
        """Writes vertices to ``fout``.

        Args:
            fout (TextIOWrapper): The file descriptor to write to
            precision (int): The number of decimal places to write (None implies write max precision)
        """
        if precision is not None:
            format_str = f'v {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}\n'
        else:
            format_str = 'v {} {} {}\n'
        for v in self.vertices:
            fout.write(format_str.format(*v))

    def _write_vt(self, fout: TextIOWrapper, precision: int) -> None:
        """Writes texture coordinates to ``fout``.

        Args:
            fout (TextIOWrapper): The file descriptor to write to
            precision (int): The number of decimal places to write (None implies write max precision)
        """
        if precision is not None:
            format_str = f'vt {{:.{precision}f}} {{:.{precision}f}}\n'
        else:
            format_str = 'vt {} {}\n'
        for vt in self.tex_coords:
            fout.write(format_str.format(*vt))

    def _write_vn(self, fout: TextIOWrapper, precision: int) -> None:
        """Writes vertex normals to ``fout``.

        Args:
            fout (TextIOWrapper): The file descriptor to write to
            precision (int): The number of decimal places to write (None implies write max precision)
        """
        if precision is not None:
            format_str = f'vn {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}\n'
        else:
            format_str = 'vn {} {} {}\n'
        for vn in self.normals:
            fout.write(format_str.format(*vn))

    def _write_f(self, fout: TextIOWrapper) -> None:
        """Writes faces and mtllib/usemtl to ``fout``.

        Args:
            fout (TextIOWrapper): The file descriptor to write to
        """
        i = 0
        mtllib, mtl = '', ''
        mat_gen = (mat for mat in self.materials)

        vert_pattern = '{}'
        if self.__have_norm:
            vert_pattern += '/{}/{}'
        elif self.__have_tex:
            vert_pattern += '/{}{}'  # Second brace will always be filled with ''
        else:
            vert_pattern += '{}{}'  # Second and third braces always filled with ''

        while i < len(self.faces):
            stop = len(self.faces)
            matlines = []
            try:
                next_mat = next(mat_gen)
                if next_mat[0] != mtllib:
                    mtllib = next_mat[0]
                    matlines.append(f'mtllib {str(mtllib.relative)}\n')
                if next_mat[1] != mtl:
                    mtl = next_mat[1]
                    matlines.append(f'usemtl {mtl}\n')
                stop = next_mat[2]
            except StopIteration:
                pass

            # First write out the faces of the lines for the current mat
            for j in range(i, stop):
                face = map(lambda x: str(x) if x != Obj.INV_IDX else '', self.faces[j].reshape((9,)))
                fout.write(f'f {vert_pattern} {vert_pattern} {vert_pattern}\n'.format(*face))

            # Then write the mtllib and usemtl lines for the next material in the list
            fout.write(''.join(matlines))
            i = stop

    def _copy_materials(self, base: Path) -> None:
        """Copies materials referenced by Obj to the destination.

        Args:
            base (Path): The base directory to copy materials into
        """
        mtllibs = {m[0] for m in self.materials}
        for mtllib in mtllibs:
            with open(mtllib.base / mtllib.relative, 'r') as m:
                content = m.read()
                materials = [line.rstrip()[len('map_Kd '):] for line in content.split('\n')
                             if line.startswith('map_Kd')]
                for material in materials:
                    mat_src = (mtllib.base / mtllib.relative).parent / material
                    mat_dst = (base / mtllib.relative).parent / material
                    os.makedirs(mat_dst.parent, exist_ok=True)
                    shutil.copy(mat_src, mat_dst)

                mtllib_dst = base / mtllib.relative
                os.makedirs(mtllib_dst.parent, exist_ok=True)
                with open(mtllib_dst, 'w+') as n:
                    n.write(content)

    def combine(self, other: 'Obj') -> None:
        """Combines another Obj into the calling Obj.
        This version will also group faces that are using the same material.

        Args:
            other (Obj): The Obj to combine

        Raises:
            RuntimeError: If ``other`` is not an Obj or if it has
                a different coordinate reference system.
        """
        if not isinstance(other, Obj):
            raise RuntimeError(f"Cannot combine {type(self)} with type {type(other)}")

        if self._crs != other._crs:  # pylint: disable=protected-access
            # TODO(rhite): convert other to self automatically?
            raise RuntimeError(f"Cannot combine objs with differing coordinate systems")

        if not self.__root:
            # Started with an empty obj that didn't call read(...)
            self.__root = other.__root  # pylint: disable=protected-access

        o_v_count = np.uint32(len(self.vertices))
        o_vt_count = np.uint32(len(self.tex_coords))
        o_vn_count = np.uint32(len(self.normals))
        o_f_count = np.uint32(len(self.faces))

        self.vertices = np.concatenate((self.vertices, other.vertices))
        self.tex_coords = np.concatenate((self.tex_coords, other.tex_coords))
        self.normals = np.concatenate((self.normals, other.normals))

        # self.faces = np.concatenate((self.faces, other.faces + [o_v_count, o_vt_count, o_vn_count]))
        # self.materials = self.materials + [(m[0], m[1], m[2] + o_f_count) for m in other.materials]

        self_mtl_names = [n for _, n, _ in self.materials]
        for cur_i in range(len(other.materials)):
            o_mtl_lib, o_mtl_name, o_mtl_face_i = other.materials[cur_i]
            o_mtl_face_j = other.materials[cur_i + 1][2] if cur_i + 1 < len(other.materials) else len(other.faces)

            if o_mtl_name in self_mtl_names:
                mtl_index = self_mtl_names.index(o_mtl_name)
                self_mtl_i = self.materials[mtl_index][2]

                self.faces = np.concatenate((self.faces[:self_mtl_i],
                                             other.faces[o_mtl_face_i:o_mtl_face_j] + [o_v_count, o_vt_count,
                                                                                       o_vn_count],
                                             self.faces[self_mtl_i:]))
                for i in range(mtl_index + 1, len(self.materials)):
                    self.materials[i][2] = self.materials[i][2] + (o_mtl_face_j - o_mtl_face_i)
            else:
                self.faces = np.concatenate(self.faces,
                                            other.faces[o_mtl_face_i:o_mtl_face_j] + [o_v_count, o_vt_count,
                                                                                      o_vn_count])
                self.materials.append((o_mtl_lib, o_mtl_name, o_mtl_face_i + o_f_count))

        self.__have_tex |= other.__have_tex  # pylint: disable=protected-access
        self.__have_norm |= other.__have_norm  # pylint: disable=protected-access

    def get_crs(self) -> CoordinateReferenceSystem:
        """Gets the coordinate system of the Obj

        Returns:
            CoordinateReferenceSystem
        """
        return self._crs

    def set_crs(self, crs: CoordinateReferenceSystem) -> None:
        """Sets the coordinate system of the Obj

        Args:
            crs (CoordinateReferenceSystem): The coordinate system in which
                the Obj's vertices live
        """
        self._crs = crs

    def convert(self, to_crs: CoordinateReferenceSystem) -> None:
        """Converts the Obj's vertices to the target coordinate system.

        Warn:
            Does not convert vertex normals

        Args:
            to_crs (CoordinateReferenceSystem): The target coordinate system to convert to
        """
        if self._crs != to_crs:
            self.vertices = CoordinateConverter(self._crs, to_crs)(self.vertices)
            self._crs = to_crs
            # TODO(rhite): Handle normals

    def __record_mat_change(self, mtllib, mtl, f):
        """Tracks indices at which materials change
        """
        if mtllib and mtl:
            mtllib = Obj.MtlLib(self.__root, mtllib)
            if len(self.materials) > 0 and self.materials[-1][2] == f:
                self.materials[-1] = [mtllib, mtl, f]
            else:
                self.materials.append([mtllib, mtl, f])

    def get_native_extents(self) -> Extents:
        """Gets the min and max along the x-, y-, and z- axis.

        Returns:
            Extents: The extents of the obj in its native coordinate space.
        """
        local_min, local_max = np.min(self.vertices, axis=0).reshape((3,)), np.max(self.vertices, axis=0).reshape((3,))
        x, y, z = zip(local_min, local_max)
        return Extents(*x, *y, *z)

    def reverse_faces(self) -> None:
        """Reverses the winding of faces by revercing "f" lines.
        """
        # Faces are stored as arrays of arrays of coordinate types
        # i.e. the 0th axis is faces, 1st axis is face points,
        #   2nd axis is the position, texture, and normal of each face point
        # You can reverse face winding by reversing the 1st axis
        self.faces = np.flip(self.faces, axis=1)


class SelfGeoreferencingObj(Obj):
    """An implementation of Obj which attempts to georeference itself.

    On read, it will attempt to parse a pyfite compatible CRS string in a comment.
    On write, it will first write a comment of the pyfite compatible CRS string of the Obj.
    """

    def write(self, dest: Union[str, Path, TextIOWrapper], precision: Union[int, Tuple[int, int, int]] = None) -> None:
        """Write the obj with a pyfite compatible CRS string as a comment.

        Refer to ``Obj`` for parameter descriptions.
        """
        if isinstance(dest, (str, Path)):
            dest = open(dest, 'w', 8192)
            if self._crs is not None:
                dest.write('# {}\n'.format(str(self._crs)))

        super().write(dest, precision)
        dest.close()

    def _custom_line_processing(self, line: str) -> None:
        """Attempts to parse any comments as a ``CoordinateReferenceSystem`` string

        Refer to ``Obj._custom_line_processing`` for parameter descriptions.
        """
        if not self._crs and line.startswith('#'):
            crs_str = CoordinateReferenceSystem.find_str(line)
            if crs_str:
                self._crs = CoordinateReferenceSystem.from_str(crs_str)
