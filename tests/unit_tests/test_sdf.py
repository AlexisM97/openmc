"""Tests for reading and writing sensitivity data files (SDF)."""

import numpy as np
import pytest
from uncertainties.unumpy import uarray

import openmc


ENERGIES = np.array([1e-5, 0.625, 1e5, 2e7])


def test_from_coefficients_and_roundtrip(run_in_tmpdir):
    coefficients = {
        ('U235', 'total'): uarray([0.1, 0.2, 0.3], [0.01, 0.01, 0.01]),
        ('U235', 'scatter'): uarray([0.0, 0.05, 0.1], [0.0, 0.01, 0.01]),
        ('U235', 'elastic'): uarray([0.0, 0.04, 0.08], [0.0, 0.01, 0.01]),
        ('U235', 'capture'): uarray([-0.1, -0.02, 0.0], [0.01, 0.0, 0.0]),
        ('U235', 'fission'): uarray([0.2, 0.17, 0.2], [0.01, 0.01, 0.01]),
        ('U235', '(n,gamma)'): uarray([-0.1, -0.02, 0.0], [0.01, 0.0, 0.0]),
        ('U235', '(n,2n)'): np.zeros(3),
        ('U235', 'nu'): uarray([0.3, 0.3, 0.4], [0.01, 0.01, 0.01]),
        ('U235', 'chi-prompt-constrained'): uarray([0.01, 0.02, -0.03],
                                                   [0.001, 0.001, 0.001]),
        ('U235', 'chi-delayed-constrained'): uarray([0.001, 0.0, -0.001],
                                                    [0.0, 0.0, 0.0]),
        ('C0', 'total'): uarray([0.01, 0.0, -0.002], [0.001, 0.0, 0.001]),
    }
    keff = openmc.statepoint.ufloat(1.001, 0.0002)
    sdf = openmc.SensitivityDataFile.from_coefficients(
        coefficients, ENERGIES, keff, 'test case')

    names = [(p.nuclide, p.reaction) for p in sdf.profiles]
    assert ('c', 'total') in names
    assert ('u-235', 'n,gamma') in names
    # Profiles that are zero are omitted unless they are always written
    assert ('u-235', 'n,2n') not in names
    chi = sdf.get('U235', 'chi')
    assert chi.mt == 1018
    assert np.allclose(chi.values, [0.011, 0.02, -0.031])
    assert sdf.get('u-235', 'fission').za == 92235
    assert sdf.get('C0', 'total').za == 6000

    sdf.export_to_file('test.sdf')
    new = openmc.SensitivityDataFile.from_file('test.sdf')
    assert new.title == 'test case'
    assert new.num_groups == 3
    assert np.allclose(new.energies, ENERGIES)
    assert new.keff == pytest.approx(1.001)
    assert new.keff_std_dev == pytest.approx(0.0002)
    assert len(new.profiles) == len(sdf.profiles)
    for a, b in zip(sdf.profiles, new.profiles):
        assert (a.nuclide, a.reaction, a.za, a.mt) == \
            (b.nuclide, b.reaction, b.za, b.mt)
        assert np.allclose(a.values, b.values, rtol=1e-6)
        assert np.allclose(a.std_dev, b.std_dev, rtol=1e-6)


def test_integrated_fields(run_in_tmpdir):
    """The integrated values follow the TSUNAMI convention: integral, its
    standard deviation, sum of absolute values, and the sum of the values with
    a sign opposite to the integral with its standard deviation."""
    profile = openmc.SensitivityProfile(
        'u-235', 'total', 92235, 1, np.array([0.5, -0.1, 0.2]),
        np.array([0.03, 0.04, 0.0]))
    sdf = openmc.SensitivityDataFile('t', ENERGIES, 1.0, 0.0, [profile])
    sdf.export_to_file('t.sdf')
    lines = open('t.sdf').read().splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith('u-235'))
    fields = [float(x) for x in lines[i + 3].split()]
    assert fields == pytest.approx([0.6, 0.05, 0.8, -0.1, 0.04])
    # Values are written by decreasing energy
    assert float(lines[i + 4].split()[0]) == pytest.approx(0.2)


def test_nuclide_names():
    from openmc.sdf import _openmc_nuclide_name, _sdf_nuclide_name
    assert _sdf_nuclide_name('U235') == ('u-235', 92235)
    assert _sdf_nuclide_name('Fe0') == ('fe', 26000)
    assert _openmc_nuclide_name('u-235') == 'U235'
    assert _openmc_nuclide_name('fe') == 'Fe0'
    assert _openmc_nuclide_name('h-1') == 'H1'


def test_chi_convention():
    coefficients = {
        ('U235', 'chi-prompt'): np.array([0.3, 0.5, 0.15]),
        ('U235', 'chi-delayed'): np.array([0.01, 0.0, 0.0]),
        ('U235', 'chi-prompt-constrained'): np.array([0.1, 0.0, -0.1]),
    }
    c = openmc.SensitivityDataFile.from_coefficients(coefficients, ENERGIES)
    u = openmc.SensitivityDataFile.from_coefficients(
        coefficients, ENERGIES, chi='unconstrained')
    assert c.get('U235', 'chi').integral == pytest.approx(0.0)
    assert u.get('U235', 'chi').integral == pytest.approx(0.96)
    with pytest.raises(ValueError):
        openmc.SensitivityDataFile.from_coefficients(
            coefficients, ENERGIES, chi='other')
