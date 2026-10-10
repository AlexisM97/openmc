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
        coefficients, ENERGIES, keff, 'test case', chi='constrained')

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
    c = openmc.SensitivityDataFile.from_coefficients(
        coefficients, ENERGIES, chi='constrained')
    u = openmc.SensitivityDataFile.from_coefficients(coefficients, ENERGIES)
    assert c.get('U235', 'chi').integral == pytest.approx(0.0)
    assert u.get('U235', 'chi').integral == pytest.approx(0.96)
    with pytest.raises(ValueError):
        openmc.SensitivityDataFile.from_coefficients(
            coefficients, ENERGIES, chi='other')


def _file(profiles, energies=ENERGIES):
    """Sensitivity data file from {(nuclide, mt): (values, std)}"""
    from openmc.sdf import _sdf_nuclide_name
    sdf = openmc.SensitivityDataFile('t', np.asarray(energies), 1.0)
    for (nuc, mt), (values, std) in profiles.items():
        name, za = _sdf_nuclide_name(nuc)
        sdf.profiles.append(openmc.SensitivityProfile(
            name, str(mt), za, mt, np.array(values, float),
            np.array(std, float)))
    return sdf


def _covariances(energies=ENERGIES):
    from openmc.data import MultigroupCovariance
    rng = np.random.default_rng(0)
    out = {}
    for nuc in ('U235', 'H1'):
        mts = (2, 18, 102) if nuc == 'U235' else (2, 102)
        n = len(energies) - 1
        a = rng.normal(size=(len(mts) * n, len(mts) * n))
        full = 0.01 * a @ a.T
        cov = {(m1, m2): full[i*n:(i + 1)*n, j*n:(j + 1)*n]
               for i, m1 in enumerate(mts) for j, m2 in enumerate(mts)
               if i <= j}
        out[nuc] = MultigroupCovariance(nuc, energies, cov)
    return out


APP = {('U235', 2): ([0.01, 0.02, 0.05], [0.001, 0.001, 0.002]),
       ('U235', 18): ([0.1, 0.2, 0.3], [0.01, 0.01, 0.01]),
       ('U235', 102): ([-0.05, -0.02, 0.0], [0.002, 0.001, 0.0]),
       ('U235', 0): ([0.01, 0.02, 0.05], [0.001, 0.001, 0.002]),
       ('U235', 101): ([-0.05, -0.02, 0.0], [0.002, 0.001, 0.0]),
       ('H1', 2): ([0.2, 0.05, 0.01], [0.01, 0.01, 0.001]),
       ('H1', 0): ([0.2, 0.05, 0.01], [0.01, 0.01, 0.001])}
EXP = {('U235', 2): ([0.02, -0.01, 0.04], [0.001, 0.001, 0.002]),
       ('U235', 18): ([0.15, 0.1, 0.3], [0.01, 0.01, 0.01]),
       ('U235', 102): ([-0.03, -0.04, 0.01], [0.002, 0.001, 0.001]),
       ('U235', 0): ([0.02, -0.01, 0.04], [0.001, 0.001, 0.002]),
       ('U235', 101): ([-0.03, -0.04, 0.01], [0.002, 0.001, 0.001]),
       ('O16', 2): ([0.03, 0.01, 0.0], [0.001, 0.001, 0.0]),
       ('O16', 0): ([0.03, 0.01, 0.0], [0.001, 0.001, 0.0])}


def test_similarity_identical():
    app = _file(APP)
    s = app.similarity(app, _covariances())
    assert s.c_k.n == pytest.approx(1.0)
    assert s.E.n == pytest.approx(1.0)
    assert s.G.n == pytest.approx(1.0)
    for part in ('fission', 'capture', 'scatter'):
        assert s.E_parts[part].n == pytest.approx(1.0)
        assert s.G_parts[part].n == pytest.approx(1.0)
    assert s.application_uncertainty == pytest.approx(
        s.experiment_uncertainty)


def test_similarity_definitions():
    app, exp = _file(APP), _file(EXP)
    cov = _covariances()
    s = app.similarity(exp, cov)

    def vec(profiles, keys):
        return np.concatenate([np.array(profiles[k][0]) if k in profiles
                               else np.zeros(3) for k in keys])

    # c_k from the sandwich rule with the block matrix of each nuclide
    var = {}
    for x, y in (('a', 'a'), ('a', 'e'), ('e', 'e')):
        total = 0.0
        for nuc, mts in (('U235', (2, 18, 102)), ('H1', (2, 102))):
            keys = [(nuc, mt) for mt in mts]
            m = np.block([[cov[nuc].get(m1, m2) for m2 in mts]
                          for m1 in mts])
            px, py = (APP if x == 'a' else EXP), (APP if y == 'a' else EXP)
            total += vec(px, keys) @ m @ vec(py, keys)
        var[x, y] = total
    assert s.c_k.n == pytest.approx(
        var['a', 'e'] / np.sqrt(var['a', 'a'] * var['e', 'e']))
    assert s.application_uncertainty == pytest.approx(np.sqrt(var['a', 'a']))

    # E is the cosine over all profiles except MT 0, 1, and 101, including
    # profiles found in only one of the files
    keys = [k for k in set(APP) | set(EXP) if k[1] not in (0, 1, 101)]
    a, e = vec(APP, keys), vec(EXP, keys)
    assert s.E.n == pytest.approx(a @ e / np.linalg.norm(a) /
                                  np.linalg.norm(e))
    keys = [('U235', 18)]
    a, e = vec(APP, keys), vec(EXP, keys)
    assert s.E_parts['fission'].n == pytest.approx(
        a @ e / np.linalg.norm(a) / np.linalg.norm(e))

    # G: differences limited to the application magnitude when signs agree
    # and the full application value when they differ
    d_fis = [0.0, 0.1, 0.0]
    d_cap = [0.02, 0.0, 0.0]
    d_sct = [0.0, 0.02, 0.01] + [0.2, 0.05, 0.01]
    total = 0.6 + 0.07 + 0.08 + 0.26
    assert s.G.n == pytest.approx(
        1.0 - (sum(d_fis) + sum(d_cap) + sum(d_sct)) / total)
    assert s.G_parts['fission'].n == pytest.approx(1.0 - 0.1 / 0.6)
    assert s.G_parts['capture'].n == pytest.approx(1.0 - 0.02 / 0.07)
    assert s.G_parts['scatter'].n == pytest.approx(1.0 - 0.29 / 0.34)


def test_similarity_uncertainty():
    """First-order uncertainties agree with the spread of the indices when
    the sensitivities are sampled within their uncertainties"""
    app, exp = _file(APP), _file(EXP)
    cov = _covariances()
    s = app.similarity(exp, cov)
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(2000):
        a = {k: (rng.normal(v, sd), sd) for k, (v, sd) in APP.items()}
        e = {k: (rng.normal(v, sd), sd) for k, (v, sd) in EXP.items()}
        samples.append(_file(a).similarity(_file(e), cov))
    for get in (lambda x: x.c_k, lambda x: x.E, lambda x: x.G):
        spread = np.std([get(x).n for x in samples])
        assert get(s).s == pytest.approx(spread, rel=0.15)


def test_similarity_group_structures():
    """Sensitivities are converted to the group structure of the covariances
    in proportion to lethargy"""
    from openmc.sdf import _lethargy_transfer
    fine = np.array([1e-5, 1e-2, 0.625, 1e3, 1e5, 1e6, 2e7])
    t = _lethargy_transfer(fine, ENERGIES)
    assert t.shape == (3, 6)
    assert np.allclose(t.sum(axis=0), 1.0)
    # Group 1e3-1e5 eV lies in the coarse group 0.625-1e5 eV
    assert t[1, 3] == pytest.approx(1.0)
    # The coarse boundary at 0.625 eV is also a fine boundary
    assert t[0, 1] == pytest.approx(1.0)
    split = _lethargy_transfer(np.array([1.0, 100.0]),
                               np.array([1.0, 10.0, 100.0]))
    assert np.allclose(split[:, 0], [0.5, 0.5])

    rng = np.random.default_rng(2)
    profiles = {k: (rng.normal(size=6), np.zeros(6)) for k in APP}
    other = {k: (rng.normal(size=6), np.zeros(6)) for k in EXP}
    a, e = _file(profiles, fine), _file(other, fine)
    s = a.similarity(e, _covariances())
    coarse = {k: (t @ v, sd[:3]) for k, (v, sd) in profiles.items()}
    coarse_e = {k: (t @ v, sd[:3]) for k, (v, sd) in other.items()}
    ref = _file(coarse).similarity(_file(coarse_e), _covariances())
    assert s.c_k.n == pytest.approx(ref.c_k.n)
    assert s.E.n == pytest.approx(ref.E.n)

    with pytest.raises(ValueError):
        a.similarity(_file(APP), _covariances())


def test_similarity_covariance_keys():
    """Covariances given by ZA take precedence, and SCALE identifiers of
    nuclides in particular materials only match covariances given by ZA"""
    cov = _covariances()
    app = _file(APP)
    by_za = {92235: cov['U235'], 'H1': cov['H1']}
    assert app.similarity(app, by_za).application_uncertainty == \
        pytest.approx(app.similarity(app, cov).application_uncertainty)

    special = _file(APP)
    for p in special.profiles:
        if p.za == 1001:
            p.za = 3001001
    u = special.similarity(special, cov).application_uncertainty
    only_u235 = app.similarity(app, {'U235': cov['U235']})
    assert u == pytest.approx(only_u235.application_uncertainty)
    u = special.similarity(special, {**cov, 3001001: cov['H1']})
    assert u.application_uncertainty == pytest.approx(
        app.similarity(app, cov).application_uncertainty)

    # Without covariances, only E and G are computed
    s = app.similarity(_file(EXP))
    assert s.c_k is None and s.application_uncertainty is None


def test_duplicate_profiles():
    sdf = _file(APP)
    sdf.profiles.append(sdf.profiles[1])
    with pytest.warns(UserWarning, match='several profiles'):
        sdf.similarity(_file(APP))


def test_read_joined_za(run_in_tmpdir):
    """ZA identifiers that fill their field run into the reaction name"""
    profile = openmc.SensitivityProfile(
        'h-1', 'scatter', 3001001, 0, np.array([0.5, -0.1, 0.2]),
        np.zeros(3))
    sdf = openmc.SensitivityDataFile('t', ENERGIES, 1.0, 0.0, [profile])
    sdf.export_to_file('t.sdf')
    text = open('t.sdf').read().replace('h-1          scatter          '
                                        '3001001', 'h-1    scatter3001001')
    open('t.sdf', 'w').write(text)
    p = openmc.SensitivityDataFile.from_file('t.sdf').profiles[0]
    assert (p.nuclide, p.reaction, p.za, p.mt) == ('h-1', 'scatter',
                                                    3001001, 0)


def test_fewer_profiles_than_header(run_in_tmpdir):
    """Files whose header overstates the number of profiles are read with a
    warning"""
    profile = openmc.SensitivityProfile(
        'u-235', 'total', 92235, 1, np.array([0.5, -0.1, 0.2]), np.zeros(3))
    sdf = openmc.SensitivityDataFile('t', ENERGIES, 1.0, 0.0, [profile])
    sdf.export_to_file('t.sdf')
    lines = open('t.sdf').read().splitlines()
    lines[2] = lines[2].replace('         1', '         2', 1)
    lines += ['', ' file verification information', ' code system: SCALE']
    open('t.sdf', 'w').write('\n'.join(lines) + '\n')
    with pytest.warns(UserWarning, match='header states 2'):
        new = openmc.SensitivityDataFile.from_file('t.sdf')
    assert len(new.profiles) == 1
