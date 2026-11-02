"""Tests of mesh universes: tetrahedralization of elements, tracking against
an equivalent CSG model, boundary sets, element filter and element data in
openmc.lib"""
import numpy as np
import pytest

import openmc
import openmc.lib
from openmc.mesh_universe import tetrahedralize, write_mesh_file

L, N = 10.0, 3


def tet_volumes(nodes, tets):
    a = nodes[tets[:, 0]]
    return np.einsum('ij,ij->i', nodes[tets[:, 1]] - a,
                     np.cross(nodes[tets[:, 2]] - a, nodes[tets[:, 3]] - a)) / 6


def box_mesh(n=N, length=L):
    """Nodes and HEX8 connectivity of [-length, length]^3 in n^3 elements,
    with shuffled node numbering"""
    g = np.linspace(-length, length, n + 1)
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    perm = np.random.default_rng(1).permutation(len(nodes))
    new = np.empty_like(perm)
    new[perm] = np.arange(len(perm))
    idx = lambda i, j, k: new[(i * (n + 1) + j) * (n + 1) + k]
    hexes = np.array([[idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k),
                       idx(i, j + 1, k), idx(i, j, k + 1), idx(i + 1, j, k + 1),
                       idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
                      for i in range(n) for j in range(n) for k in range(n)])
    return nodes[perm], hexes


def test_tetrahedralize_conforming():
    nodes, hexes = box_mesh()
    tets, parent = tetrahedralize(hexes, 'HEX8')
    vol = np.abs(tet_volumes(nodes, tets))
    assert vol.min() > 0.0
    assert np.bincount(parent, vol) == pytest.approx((2 * L / N)**3)
    # Interior faces are shared by exactly two tetrahedra, boundary faces of
    # the box by one
    faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]
                    .reshape(-1, 3), axis=1)
    _, count = np.unique(faces, axis=0, return_counts=True)
    assert set(count) == {1, 2}
    assert (count == 1).sum() == 6 * N**2 * 2


@pytest.mark.parametrize('kind, n_nodes', [('WEDGE6', 6), ('PYRAMID5', 5),
                                           ('TET4', 4)])
def test_tetrahedralize_volumes(kind, n_nodes):
    nodes = {
        'WEDGE6': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                   [0, 1, 1]],
        'PYRAMID5': [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [.5, .5, 1]],
        'TET4': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]}[kind]
    volume = {'WEDGE6': 0.5, 'PYRAMID5': 1 / 3, 'TET4': 1 / 6}[kind]
    nodes = np.array(nodes, dtype=float)
    for perm in np.random.default_rng(2).permuted(
            np.tile(np.arange(n_nodes), (5, 1)), axis=1):
        # Renumber the nodes; the element keeps its local ordering
        renumbered = np.empty(n_nodes, dtype=int)
        renumbered[np.arange(n_nodes)] = perm
        x = np.empty_like(nodes)
        x[perm] = nodes
        tets, _ = tetrahedralize(renumbered[None, :], kind)
        assert np.abs(tet_volumes(x, tets)).sum() == pytest.approx(volume)


@pytest.fixture
def fuel():
    mat = openmc.Material()
    mat.add_nuclide('U235', 0.06)
    mat.add_nuclide('U238', 0.94)
    mat.add_nuclide('H1', 12.0)
    mat.add_nuclide('O16', 8.0)
    mat.set_density('g/cm3', 2.0)
    return mat


def settings(particles=2000):
    s = openmc.Settings()
    s.particles = particles
    s.inactive = 5
    s.batches = 25
    s.seed = 1
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box([-L / 2] * 3, [L / 2] * 3))
    return s


def mesh_model(fuel, reflect=False):
    """Two-block box mesh (z < 0 and z > 0) with side set 7 on x = -L"""
    nodes, hexes = box_mesh()
    centre_z = nodes[hexes, 2].mean(axis=1)
    order = np.argsort(centre_z > 0, kind='stable')
    hexes = hexes[order]
    top = centre_z[order] > 0
    blocks = [{'id': 1, 'type': 'HEX8', 'connectivity': hexes[~top]},
              {'id': 2, 'type': 'HEX8', 'connectivity': hexes[top]}]
    xmin = np.isclose(nodes[hexes[:, 0], 0], -L)
    side_sets = [{'id': 7, 'elements': np.nonzero(xmin)[0],
                  'sides': np.full(xmin.sum(), 3)}]
    info = write_mesh_file('box.h5', nodes, blocks, side_sets)
    assert info == {'nodes': (N + 1)**3, 'elements': N**3, 'tets': 6 * N**3}
    cells = [openmc.MeshCell(1, fuel), openmc.MeshCell(2, fuel)]
    univ = openmc.MeshUniverse('box.h5', boundary='vacuum',
                               boundary_sets={7: 'reflective'} if reflect else {})
    univ.add_cells(cells)
    return univ, cells


def csg_model(fuel, reflect=False):
    x0 = openmc.XPlane(-L, boundary_type='reflective' if reflect else 'vacuum')
    others = [openmc.XPlane(L), openmc.YPlane(-L), openmc.YPlane(L),
              openmc.ZPlane(-L), openmc.ZPlane(L)]
    for s in others:
        s.boundary_type = 'vacuum'
    box = +x0 & -others[0] & +others[1] & -others[2]
    z0 = openmc.ZPlane(0.0)
    return [openmc.Cell(fill=fuel, region=box & +others[3] & -z0),
            openmc.Cell(fill=fuel, region=box & +z0 & -others[4])]


@pytest.mark.parametrize('reflect', [False, True])
def test_mesh_vs_csg(run_in_tmpdir, fuel, reflect):
    k = {}
    for kind in ('mesh', 'csg'):
        if kind == 'mesh':
            univ, cells = mesh_model(fuel, reflect)
            geometry = openmc.Geometry(univ)
        else:
            cells = csg_model(fuel, reflect)
            geometry = openmc.Geometry(cells)
        model = openmc.Model(geometry, openmc.Materials([fuel]), settings())
        sp = model.run()
        with openmc.StatePoint(sp) as f:
            k[kind] = f.keff
    diff = k['mesh'] - k['csg']
    assert abs(diff.n) < 4 * diff.s


def test_summary(run_in_tmpdir, fuel):
    univ, cells = mesh_model(fuel)
    model = openmc.Model(openmc.Geometry(univ), openmc.Materials([fuel]),
                         settings(200))
    model.settings.batches = 6
    sp = model.run()
    with openmc.StatePoint(sp) as f:
        root = f.summary.geometry.root_universe
    assert isinstance(root, openmc.MeshUniverse)
    assert {c.block for c in root.cells.values()} == {1, 2}


def test_lib_element_data(run_in_tmpdir, fuel):
    univ, cells = mesh_model(fuel)
    elements = openmc.Tally()
    elements.filters = [openmc.ElementFilter(univ)]
    elements.scores = ['fission', 'flux']
    blocks = openmc.Tally()
    blocks.filters = [openmc.CellFilter(cells)]
    blocks.scores = ['fission', 'flux']
    model = openmc.Model(openmc.Geometry(univ), openmc.Materials([fuel]),
                         settings(1000), openmc.Tallies([elements, blocks]))
    model.export_to_model_xml()

    openmc.lib.init(output=False)
    try:
        mesh = openmc.lib.MeshUniverse(univ.id)
        assert mesh.n_elements == N**3
        assert mesh.volumes.sum() == pytest.approx((2 * L)**3)
        assert sorted(np.unique(mesh.blocks)) == [1, 2]
        T = mesh.temperatures
        assert T == pytest.approx(np.full(N**3, T[0]))
        mesh.temperatures = T
        with pytest.raises(openmc.lib.exc.OpenMCError):
            mesh.density_multipliers = np.full(N**3, -1.0)
        with pytest.raises(ValueError):
            mesh.temperatures = T[:-1]
        mesh.density_multipliers = np.linspace(0.5, 1.0, N**3)
        assert mesh.density_multipliers[0] == pytest.approx(0.5)
        openmc.lib.run(output=False)
        per_element = openmc.lib.tallies[elements.id].mean.reshape(N**3, 2)
        per_block = openmc.lib.tallies[blocks.id].mean.reshape(2, 2)
    finally:
        openmc.lib.finalize()
    # The elements partition the blocks
    assert per_element.sum(axis=0) == pytest.approx(per_block.sum(axis=0))
    assert (per_element[:, 1] > 0).all()
