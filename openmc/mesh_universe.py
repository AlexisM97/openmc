"""Universes whose geometry is an unstructured mesh, for tracking particles on
the mesh of a coupled physics code (for instance a MOOSE heat conduction
mesh), with temperatures and densities per element.

OpenMC tracks particles on linear tetrahedra. Other elements (hexahedra,
wedges, pyramids) are split into tetrahedra so that the faces shared by two
elements are split in the same way: each face is triangulated as a fan from
its node with the lowest index, and each element as cones from its node with
the lowest index to the triangles of the faces that do not contain it. This
holds for convex elements, the usual case for meshes of solids. The
tetrahedra keep the index of their element, so that element temperatures and
tallies refer to the elements of the original mesh.
"""
from __future__ import annotations

from numbers import Real
from pathlib import Path
import lxml.etree as ET

import h5py
import numpy as np

import openmc
import openmc.checkvalue as cv
from openmc.bounding_box import BoundingBox

# Faces of the element types (Exodus side numbering), local node indices
_FACES = {
    'TET4': [[0, 1, 3], [1, 2, 3], [0, 2, 3], [0, 1, 2]],
    'PYRAMID5': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2, 1]],
    'WEDGE6': [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2], [0, 2, 1],
               [3, 4, 5]],
    'HEX8': [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 4, 7, 3],
             [0, 3, 2, 1], [4, 5, 6, 7]],
}
_N_CORNERS = {'TET4': 4, 'PYRAMID5': 5, 'WEDGE6': 6, 'HEX8': 8}


def _element_type(name: str) -> str:
    """Linear element type of an Exodus element type (corner nodes only)"""
    name = name.upper()
    for prefix, linear in (('TET', 'TET4'), ('PYRAMID', 'PYRAMID5'),
                           ('WEDGE', 'WEDGE6'), ('HEX', 'HEX8')):
        if name.startswith(prefix):
            return linear
    raise ValueError(f'Element type {name} is not supported by mesh universes.')


def _fan(face: np.ndarray) -> np.ndarray:
    """Triangles (n_elements, n_triangles, 3) of polygonal faces
    (n_elements, n_nodes), as fans from the node with the lowest index"""
    n = face.shape[1]
    start = face.argmin(axis=1)
    rolled = face[np.arange(len(face))[:, None],
                  (start[:, None] + np.arange(n)) % n]
    return np.stack([np.repeat(rolled[:, :1], n - 2, axis=1),
                     rolled[:, 1:-1], rolled[:, 2:]], axis=2)


def tetrahedralize(connectivity: np.ndarray, element_type: str):
    """Split elements of one type into conforming tetrahedra

    Parameters
    ----------
    connectivity : numpy.ndarray
        Node indices of the elements, shape (n_elements, n_nodes); only the
        corner nodes are used
    element_type : str
        Exodus element type, e.g. 'HEX8', 'HEX27', 'WEDGE6', 'TET4'

    Returns
    -------
    tets : numpy.ndarray
        Node indices of the tetrahedra, shape (n_tets, 4)
    parent : numpy.ndarray
        Index (in connectivity) of the element of each tetrahedron

    """
    kind = _element_type(element_type)
    conn = np.asarray(connectivity)[:, :_N_CORNERS[kind]]
    if kind == 'TET4':
        return conn.copy(), np.arange(len(conn))
    apex = conn.min(axis=1)
    tets, parent = [], []
    for face in _FACES[kind]:
        tri = _fan(conn[:, face])                  # (n, m, 3)
        for j in range(tri.shape[1]):
            t = tri[:, j]
            keep = ~(t == apex[:, None]).any(axis=1)
            tets.append(np.column_stack([apex[keep], t[keep]]))
            parent.append(np.nonzero(keep)[0])
    return np.concatenate(tets), np.concatenate(parent)


def _face_keys(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    s = np.sort(tri, axis=-1).astype(np.int64)
    return (s[..., 0] * n_nodes + s[..., 1]) * n_nodes + s[..., 2]


def _read_netcdf(path):
    """Variables of an Exodus file (netCDF classic or netCDF-4/HDF5)"""
    with open(path, 'rb') as fh:
        magic = fh.read(4)
    if magic[:3] == b'CDF':
        from scipy.io import netcdf_file
        f = netcdf_file(path, 'r', mmap=False)
        var = {k: v for k, v in f.variables.items()}
        get = lambda k: var[k][:].copy()
        attr = lambda k, a: getattr(var[k], a)
        return f, var, get, attr
    f = h5py.File(path, 'r')
    var = {k: f[k] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    get = lambda k: var[k][()]
    attr = lambda k, a: var[k].attrs[a]
    return f, var, get, attr


def _names(raw) -> list[str]:
    return [b''.join(np.atleast_1d(r)).decode().strip('\x00').strip()
            if np.asarray(r).dtype.kind == 'S' and np.asarray(r).ndim
            else (r.decode().strip('\x00') if isinstance(r, bytes) else str(r))
            for r in raw]


def read_exodus(path, length_multiplier: float = 1.0):
    """Read the elements, blocks and side sets of an Exodus mesh

    Parameters
    ----------
    path : path-like
        Exodus file
    length_multiplier : float
        Factor applied to the coordinates (100 for a mesh in meters)

    Returns
    -------
    dict
        'nodes' (n, 3), 'blocks' list of dicts with 'id', 'name', 'type' and
        'connectivity' (0-based), 'side_sets' list of dicts with 'id',
        'name', 'elements' (0-based global element index) and 'sides'
        (0-based local side)

    """
    f, var, get, attr = _read_netcdf(path)
    try:
        if 'coord' in var:
            nodes = np.asarray(get('coord')).T
        else:
            nodes = np.column_stack(
                [get(f'coord{c}') for c in 'xyz' if f'coord{c}' in var])
        if nodes.shape[1] == 2:
            raise ValueError('Mesh universes require a 3D mesh.')
        n_blocks = len([k for k in var if k.startswith('connect')])
        ids = get('eb_prop1') if 'eb_prop1' in var else np.arange(1, n_blocks + 1)
        names = _names(get('eb_names')) if 'eb_names' in var else [''] * n_blocks
        blocks = []
        for i in range(n_blocks):
            key = f'connect{i + 1}'
            etype = attr(key, 'elem_type')
            etype = etype.decode() if isinstance(etype, bytes) else str(etype)
            blocks.append({'id': int(ids[i]), 'name': names[i], 'type': etype,
                           'connectivity': np.asarray(get(key)).astype(np.int64) - 1})
        side_sets = []
        n_sets = len([k for k in var if k.startswith('elem_ss')])
        if n_sets:
            sids = get('ss_prop1') if 'ss_prop1' in var else np.arange(1, n_sets + 1)
            snames = _names(get('ss_names')) if 'ss_names' in var else [''] * n_sets
            for i in range(n_sets):
                side_sets.append({
                    'id': int(sids[i]), 'name': snames[i],
                    'elements': np.asarray(get(f'elem_ss{i + 1}')).astype(np.int64) - 1,
                    'sides': np.asarray(get(f'side_ss{i + 1}')).astype(np.int64) - 1})
    finally:
        f.close()
    return {'nodes': length_multiplier * np.asarray(nodes, dtype=float),
            'blocks': blocks, 'side_sets': side_sets}


def write_mesh_file(path, nodes, blocks, side_sets=()):
    """Write the tetrahedral mesh file of a mesh universe

    Parameters
    ----------
    path : path-like
        HDF5 file to write
    nodes : numpy.ndarray
        Node coordinates (n, 3) in cm
    blocks : list of dict
        Element blocks with 'id', 'type' and 'connectivity' (0-based node
        indices); elements are numbered in the order of the blocks
    side_sets : list of dict
        Boundary sets with 'id', 'elements' (0-based global element indices)
        and 'sides' (0-based local sides)

    Returns
    -------
    dict
        Numbers of nodes, elements and tetrahedra

    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    tets, tet_element, element_block, offsets = [], [], [], []
    offset = 0
    for b, block in enumerate(blocks):
        t, parent = tetrahedralize(block['connectivity'], block['type'])
        tets.append(t)
        tet_element.append(parent + offset)
        element_block.append(np.full(len(block['connectivity']), b))
        offsets.append(offset)
        offset += len(block['connectivity'])
    tets = np.concatenate(tets).astype(np.int32)
    tet_element = np.concatenate(tet_element).astype(np.int32)
    element_block = np.concatenate(element_block).astype(np.int32)

    face_sets = np.full((len(tets), 4), -1, dtype=np.int32)
    if side_sets:
        # Triangles of each side, keyed by their sorted nodes
        starts = np.array(offsets + [offset])
        keys, values = [], []
        for ss in side_sets:
            elems = np.asarray(ss['elements'])
            sides = np.asarray(ss['sides'])
            block_of = np.searchsorted(starts, elems, side='right') - 1
            for b in np.unique(block_of):
                kind = _element_type(blocks[b]['type'])
                conn = np.asarray(blocks[b]['connectivity'])
                sel = block_of == b
                local = elems[sel] - starts[b]
                for s in np.unique(sides[sel]):
                    rows = local[sides[sel] == s]
                    tri = _fan(conn[rows][:, _FACES[kind][s]])
                    keys.append(_face_keys(tri, n).ravel())
                    values.append(np.full(tri.shape[0] * tri.shape[1], ss['id']))
        keys = np.concatenate(keys)
        values = np.concatenate(values)
        order = np.argsort(keys)
        keys, values = keys[order], values[order]
        opposite = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        for k in range(4):
            fk = _face_keys(tets[:, opposite[k]], n)
            i = np.clip(np.searchsorted(keys, fk), 0, len(keys) - 1)
            hit = keys[i] == fk
            face_sets[hit, k] = values[i[hit]]

    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('mesh_universe')
        f.create_dataset('nodes', data=nodes.ravel())
        f.create_dataset('tets', data=tets.ravel())
        f.create_dataset('tet_element', data=tet_element)
        f.create_dataset('element_block', data=element_block)
        f.create_dataset('block_ids',
                         data=np.array([b['id'] for b in blocks], dtype=np.int32))
        f.create_dataset('block_names',
                         data=np.array([b.get('name', '') for b in blocks], dtype='S'))
        if side_sets:
            f.create_dataset('tet_face_sets', data=face_sets.ravel())
    return {'nodes': n, 'elements': len(element_block), 'tets': len(tets)}


def exodus_to_mesh_file(exodus_path, path, length_multiplier: float = 1.0):
    """Convert an Exodus mesh to the mesh file of a mesh universe

    Parameters
    ----------
    exodus_path : path-like
        Exodus file (e.g. a MOOSE mesh)
    path : path-like
        HDF5 mesh file to write
    length_multiplier : float
        Factor applied to the coordinates (100 for a mesh in meters)

    Returns
    -------
    dict
        Block IDs and names, side set IDs and names, and the numbers of nodes,
        elements and tetrahedra

    """
    mesh = read_exodus(exodus_path, length_multiplier)
    info = write_mesh_file(path, mesh['nodes'], mesh['blocks'], mesh['side_sets'])
    info['blocks'] = {b['id']: b['name'] for b in mesh['blocks']}
    info['side_sets'] = {s['id']: s['name'] for s in mesh['side_sets']}
    return info


class MeshCell(openmc.Cell):
    """Cell of a mesh universe: the elements of one element block

    Parameters
    ----------
    block : int
        ID of the element block
    fill : openmc.Material or None
        Material of the block (None for void)
    cell_id : int, optional
        Unique identifier of the cell
    name : str, optional
        Name of the cell

    """

    def __init__(self, block: int, fill=None, cell_id=None, name=''):
        super().__init__(cell_id, name, fill, None)
        self.block = block

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, block):
        cv.check_type('element block', block, int)
        self._block = block

    def bounding_box(self):
        return BoundingBox.infinite()

    def get_all_cells(self, memo=None):
        return {}

    def get_all_universes(self, memo=None):
        return {}

    def create_xml_subelement(self, xml_element, memo=None):
        if self.fill_type not in ('void', 'material'):
            raise TypeError('Cells of mesh universes must be filled with one '
                            'material or void.')
        element = ET.Element('cell')
        element.set('id', str(self.id))
        element.set('block', str(self.block))
        if self.name:
            element.set('name', self.name)
        element.set('material', 'void' if self.fill is None else str(self.fill.id))
        if self.temperature is not None:
            element.set('temperature', str(self.temperature))
        return element


class MeshUniverse(openmc.UniverseBase):
    """Universe whose geometry is a tetrahedral mesh file (see
    :func:`exodus_to_mesh_file`); particles are tracked through the faces of
    the tetrahedra. Each element block is one :class:`MeshCell`. Temperatures
    and density multipliers can be set per element through
    :class:`openmc.lib.MeshUniverse`.

    Parameters
    ----------
    filename : path-like
        Mesh file
    universe_id : int, optional
        Unique identifier of the universe
    name : str, optional
        Name of the universe
    boundary : str
        Boundary condition of the boundary faces of the mesh that are not in
        a boundary set given in `boundary_sets`: 'vacuum', 'reflective' or
        'transmission'
    boundary_sets : dict, optional
        Boundary condition of the faces of each side set of the mesh
    length_multiplier : float
        Factor applied to the coordinates of the mesh file

    """

    def __init__(self, filename, universe_id=None, name='', boundary='vacuum',
                 boundary_sets=None, length_multiplier=1.0):
        super().__init__(universe_id, name)
        self.filename = filename
        self.boundary = boundary
        self.boundary_sets = boundary_sets or {}
        self.length_multiplier = length_multiplier

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        cv.check_type('mesh universe filename', filename, (str, Path))
        self._filename = filename

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        cv.check_value('boundary', boundary,
                       ('vacuum', 'reflective', 'transmission'))
        self._boundary = boundary

    @property
    def length_multiplier(self):
        return self._length_multiplier

    @length_multiplier.setter
    def length_multiplier(self, mult):
        cv.check_type('length multiplier', mult, Real)
        self._length_multiplier = mult

    @property
    def bounding_box(self):
        with h5py.File(self.filename) as f:
            x = f['nodes'][()].reshape(-1, 3) * self.length_multiplier
        return BoundingBox(x.min(axis=0), x.max(axis=0))

    def add_cell(self, cell):
        if not isinstance(cell, MeshCell):
            raise TypeError(f'Cell {cell} of mesh universe {self.id} is not '
                            'a MeshCell.')
        self._cells[cell.id] = cell

    @classmethod
    def from_hdf5(cls, group, cells):
        """Create a mesh universe from its group of a summary file

        Parameters
        ----------
        group : h5py.Group
            Group of the universe
        cells : dict
            Cells by ID

        Returns
        -------
        openmc.MeshUniverse

        """
        universe_id = int(group.name.split('/')[-1].lstrip('universe '))
        universe = cls(group['filename'][()].decode(), universe_id=universe_id)
        for cell_id in group['cells'][()]:
            universe.add_cell(cells[int(cell_id)])
        return universe

    def remove_cell(self, cell):
        self._cells.pop(cell.id, None)

    def _partial_deepcopy(self):
        return MeshUniverse(self.filename, name=self.name,
                            boundary=self.boundary,
                            boundary_sets=dict(self.boundary_sets),
                            length_multiplier=self.length_multiplier)

    def create_xml_subelement(self, xml_element, memo=None):
        if memo is None:
            memo = set()
        if self in memo:
            return
        memo.add(self)
        element = ET.Element('mesh_universe')
        element.set('id', str(self.id))
        if self.name:
            element.set('name', self.name)
        element.set('filename', str(self.filename))
        element.set('boundary', self.boundary)
        if self.length_multiplier != 1.0:
            element.set('length_multiplier', str(self.length_multiplier))
        for cell in self.cells.values():
            element.append(cell.create_xml_subelement(xml_element, memo))
        for set_id, bc in self.boundary_sets.items():
            ET.SubElement(element, 'boundary', set=str(set_id), type=bc)
        xml_element.append(element)
