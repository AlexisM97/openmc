"""Tests for multigroup covariances and propagation of nuclear data
uncertainties."""

import shutil

import numpy as np
import pytest
from uncertainties.unumpy import uarray

import openmc.data
from openmc.data.covariance import CHI_MT, _primary_mts


ENERGIES = [1e-5, 1e5, 2e7]


def _cov(covariances, lumped=None, nuclide='U235'):
    return openmc.data.MultigroupCovariance(
        nuclide, ENERGIES, covariances, lumped=lumped)


def test_get_and_std_dev():
    c = _cov({(18, 18): np.diag([1e-4, 4e-4]),
              (18, 102): np.array([[1e-5, 0.0], [2e-5, 3e-5]]),
              (102, 102): np.diag([1e-2, 1e-2])})
    assert c.num_groups == 2
    assert c.mts == [18, 102]
    assert np.allclose(c.std_dev(18), [0.01, 0.02])
    assert np.allclose(c.get(102, 18), c.get(18, 102).T)
    assert c.get(2, 18) is None
    corr = c.correlation(18, 102)
    assert corr[1, 0] == pytest.approx(2e-5 / (0.02 * 0.1))


def test_hdf5_roundtrip(run_in_tmpdir):
    c = _cov({(2, 2): np.eye(2) * 1e-4, (2, 4): np.ones((2, 2)) * 1e-5},
             lumped={52: 851})
    c.cross_sections[2] = np.array([10.0, 4.0])
    c.temperature = 293.6
    c.export_to_hdf5('cov.h5', 'w')
    new = openmc.data.MultigroupCovariance.from_hdf5('cov.h5')
    assert new.nuclide == 'U235'
    assert np.allclose(new.energies, ENERGIES)
    assert set(new.covariances) == set(c.covariances)
    assert np.allclose(new.get(4, 2), c.get(4, 2))
    assert new.lumped == {52: 851}
    assert new.temperature == pytest.approx(293.6)


def test_sandwich_rule():
    cov = {(18, 18): np.diag([1e-4, 4e-4]),
           (102, 102): np.diag([1e-2, 1e-2]),
           (18, 102): np.array([[0.0, 0.0], [0.0, 1e-3]]),
           (452, 452): np.full((2, 2), 2.5e-5)}
    s_fis = np.array([0.1, 0.6])
    s_cap = np.array([-0.01, -0.03])
    s_nu = np.array([0.2, 0.8])
    coefficients = {
        ('U235', 'fission'): uarray(s_fis, [0.01, 0.01]),
        ('U235', '(n,gamma)'): s_cap,
        ('U235', 'nu'): s_nu,
        ('U235', 'nu-prompt'): s_nu,
        ('U235', 'nu-delayed'): np.zeros(2),
        ('U235', 'inelastic'): np.ones(2),
    }
    result = openmc.data.propagate_uncertainty(
        coefficients, {'U235': _cov(cov)}, ENERGIES)

    expected = (s_fis @ cov[18, 18] @ s_fis + s_cap @ cov[102, 102] @ s_cap +
                2 * s_fis @ cov[18, 102] @ s_cap + s_nu @ cov[452, 452] @ s_nu)
    assert result.variance == pytest.approx(expected)
    assert result.std_dev == pytest.approx(np.sqrt(expected))
    assert sum(c.variance for c in result.contributions) == \
        pytest.approx(expected)
    assert result.by_nuclide()['U235'] == pytest.approx(expected)
    labels = {(c.reaction1, c.reaction2) for c in result.contributions}
    assert ('(n,fission)', '(n,gamma)') in labels
    assert ('nubar', 'nubar') in labels
    # The cross term between fission and capture is negative here
    cross = [c for c in result.contributions if c.reaction2 == '(n,gamma)'
             and c.reaction1 == '(n,fission)'][0]
    assert cross.variance < 0.0
    assert result.missing_sensitivities == []
    assert result.missing_covariances == []


def test_missing_data():
    cov = {(2, 2): np.eye(2) * 1e-4, (16, 16): np.eye(2) * 1e-2}
    coefficients = {('U235', 'elastic'): np.ones(2),
                    ('U235', '(n,p)'): np.ones(2),
                    ('U238', 'fission'): np.ones(2)}
    result = openmc.data.propagate_uncertainty(
        coefficients, {'U235': _cov(cov)})
    assert result.variance == pytest.approx(2e-4)
    assert ('U235', '(n,2n)') in result.missing_sensitivities
    assert ('U235', '(n,p)') in result.missing_covariances
    assert ('U238', 'fission') in result.missing_covariances


def test_group_mismatch():
    with pytest.raises(ValueError):
        openmc.data.propagate_uncertainty(
            {('U235', 'fission'): np.ones(2)},
            {'U235': _cov({(18, 18): np.eye(2)})}, [1e-5, 1e6, 2e7])


def test_primary_reactions():
    eye = np.eye(2)
    # Inelastic levels and their lump are represented by MT 4, while a lump of
    # charged-particle levels without summary covariances is kept
    c = _cov({(mt, mt): eye for mt in (1, 2, 4, 18, 19, 51, 102, 452, 455,
                                       456, 851, 852, CHI_MT)},
             lumped={52: 851, 53: 851, 600: 852, 800: 852})
    assert _primary_mts(c) == [2, 4, 18, 102, 452, 852, CHI_MT]

    # Without MT 4 or MT 452, levels and prompt/delayed multiplicities are used
    c = _cov({(mt, mt): eye for mt in (2, 51, 455, 456, 851)},
             lumped={52: 851, 53: 851})
    assert _primary_mts(c) == [2, 51, 455, 456, 851]


def test_chi_and_lumped_sensitivities():
    cov = {(CHI_MT, CHI_MT): np.eye(2) * 1e-2,
           (851, 851): np.eye(2) * 1e-2}
    coefficients = {
        ('U235', 'chi-prompt-constrained'): np.array([0.1, -0.1]),
        ('U235', 'chi-delayed-constrained'): np.array([0.01, -0.01]),
        ('U235', '(n,n2)'): np.array([0.0, 0.1]),
        ('U235', '(n,n3)'): np.array([0.0, 0.2]),
    }
    result = openmc.data.propagate_uncertainty(
        coefficients, {'U235': _cov(cov, lumped={52: 851, 53: 851})})
    s_chi = np.array([0.11, -0.11])
    s_lump = np.array([0.0, 0.3])
    assert result.variance == pytest.approx(
        1e-2 * (s_chi @ s_chi + s_lump @ s_lump))


def _endf_line(values, mat, mf, mt, ns):
    """Format a line of an ENDF-6 file"""
    fields = ''
    for v in values:
        if isinstance(v, int):
            fields += f'{v:11d}'
        else:
            mant, exp = f'{v:.6e}'.split('e')
            fields += f'{mant}{int(exp):+d}'.rjust(11)
    return f'{fields:66s}{mat:4d}{mf:2d}{mt:3d}{ns:5d}\n'


def test_from_errorr(run_in_tmpdir):
    mat = 9228
    lines = ['errorr output' + ' ' * 53 + f'{0:4d}{0:2d}{0:3d}{0:5d}\n']
    # MF1: group structure
    lines.append(_endf_line([92235.0, 233.0, 6, 0, -11, 0], mat, 1, 451, 1))
    lines.append(_endf_line([293.6, 0.0, 2, 0, 3, 0], mat, 1, 451, 2))
    lines.append(_endf_line([1e-5, 1e5, 2e7], mat, 1, 451, 3))
    lines.append(_endf_line([], mat, 1, 0, 99999))
    # MF3: group cross sections
    lines.append(_endf_line([92235.0, 0.0, 0, 0, 2, 0], mat, 3, 18, 1))
    lines.append(_endf_line([50.0, 1.25], mat, 3, 18, 2))
    lines.append(_endf_line([], mat, 3, 0, 99999))
    # MF33: covariances of MT 18 with itself and MT 102, with the first row
    # of the MT 102 block only covering the second column
    lines.append(_endf_line([92235.0, 233.0, 0, 0, 0, 2], mat, 33, 18, 1))
    lines.append(_endf_line([0.0, 0.0, 0, 18, 0, 2], mat, 33, 18, 2))
    lines.append(_endf_line([0.0, 0.0, 2, 1, 2, 1], mat, 33, 18, 3))
    lines.append(_endf_line([1e-4, 2e-5], mat, 33, 18, 4))
    lines.append(_endf_line([0.0, 0.0, 2, 1, 2, 2], mat, 33, 18, 5))
    lines.append(_endf_line([2e-5, 3e-4], mat, 33, 18, 6))
    lines.append(_endf_line([0.0, 0.0, 0, 102, 0, 2], mat, 33, 18, 7))
    lines.append(_endf_line([0.0, 0.0, 1, 2, 1, 1], mat, 33, 18, 8))
    lines.append(_endf_line([-5e-5], mat, 33, 18, 9))
    lines.append(_endf_line([0.0, 0.0, 1, 2, 1, 2], mat, 33, 18, 10))
    lines.append(_endf_line([-6e-5], mat, 33, 18, 11))
    # MF33 section of a reaction included in a lumped covariance
    lines.append(_endf_line([92235.0, 233.0, 0, 851, 0, 0], mat, 33, 52, 1))
    with open('errorr33', 'w') as fh:
        fh.writelines(lines)

    c = openmc.data.MultigroupCovariance.from_errorr('errorr33', 'U235')
    assert np.allclose(c.energies, [1e-5, 1e5, 2e7])
    assert c.temperature == pytest.approx(293.6)
    assert np.allclose(c.cross_sections[18], [50.0, 1.25])
    assert np.allclose(c.get(18, 18), [[1e-4, 2e-5], [2e-5, 3e-4]])
    assert np.allclose(c.get(18, 102), [[0.0, -5e-5], [0.0, -6e-5]])
    assert c.lumped == {52: 851}


@pytest.mark.skipif(shutil.which('njoy') is None, reason='NJOY not available')
def test_from_endf(run_in_tmpdir, endf_data):
    filename = f'{endf_data}/neutrons/n-001_H_001.endf'
    c = openmc.data.MultigroupCovariance.from_endf(
        filename, [1e-5, 1.0, 1e6, 2e7])
    assert c.nuclide == 'H1'
    assert 2 in c.mts
    std = c.std_dev(2)
    assert np.all(std >= 0.0) and np.all(std < 1.0)
    m = c.get(2, 2)
    assert np.allclose(m, m.T)


def test_statistical_uncertainty():
    """Compare the propagated statistical uncertainty and bias of the
    variance with sampling of noisy sensitivity coefficients."""
    rng = np.random.default_rng(1)
    cov = {(18, 18): np.array([[4e-4, 1e-4], [1e-4, 9e-4]]),
           (102, 102): np.diag([1e-2, 2e-2]),
           (18, 102): np.array([[5e-4, 0.0], [2e-4, 1e-3]])}
    s_fis, sd_fis = np.array([0.3, 0.6]), np.array([0.02, 0.03])
    s_cap, sd_cap = np.array([-0.05, -0.1]), np.array([0.01, 0.005])
    coefficients = {('U235', 'fission'): uarray(s_fis, sd_fis),
                    ('U235', '(n,gamma)'): uarray(s_cap, sd_cap)}
    result = openmc.data.propagate_uncertainty(
        coefficients, {'U235': _cov(cov)})

    samples = []
    for _ in range(20000):
        sample = {
            ('U235', 'fission'): s_fis + rng.normal(0.0, sd_fis),
            ('U235', '(n,gamma)'): s_cap + rng.normal(0.0, sd_cap)}
        samples.append(openmc.data.propagate_uncertainty(
            sample, {'U235': _cov(cov)}).variance)
    samples = np.array(samples)

    assert result.variance_std_dev == pytest.approx(samples.std(), rel=0.05)
    assert samples.mean() - result.variance == pytest.approx(
        result.bias, rel=0.1)
    assert result.unbiased_variance == pytest.approx(
        result.variance - result.bias)
    assert result.std_dev_uncertainty == pytest.approx(
        result.variance_std_dev / (2 * result.std_dev))
    assert sum(c.bias for c in result.contributions) == \
        pytest.approx(result.bias)

    # Without uncertainties, nothing is propagated
    nominal = {k: np.array([v.n for v in vals])
               for k, vals in coefficients.items()}
    result = openmc.data.propagate_uncertainty(nominal, {'U235': _cov(cov)})
    assert result.variance_std_dev == 0.0
    assert result.bias == 0.0
