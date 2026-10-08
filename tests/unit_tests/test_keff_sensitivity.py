"""Test the calculation of k-effective sensitivity coefficients using the
collision history-based method."""

import lxml.etree as ET
import numpy as np
import pytest
import openmc


ENERGIES = [1e-5, 1e5, 1e6, 2e7]


def test_filter_bins():
    f = openmc.SensitivityFilter(['U235', 'U238'], ['fission', 'nu-prompt'],
                                 ENERGIES)
    assert f.shape == (2, 2, 3)
    assert f.num_bins == 12
    assert f.bins[0] == ('U235', 'fission', 1e-5, 1e5)
    assert f.bins[5] == ('U235', 'nu-prompt', 1e6, 2e7)
    assert f.bins[6] == ('U238', 'fission', 1e-5, 1e5)

    df = f.get_pandas_dataframe(f.num_bins, 1)
    assert list(df['sensitivity', 'nuclide']) == ['U235'] * 6 + ['U238'] * 6
    assert list(df['sensitivity', 'reaction'][:6]) == \
        ['fission'] * 3 + ['nu-prompt'] * 3
    assert np.allclose(df['sensitivity', 'energy low [eV]'][:3], ENERGIES[:-1])

    # Group structure given by name
    f = openmc.SensitivityFilter('U235', 'capture', 'CASMO-2')
    assert f.shape == (1, 1, 2)


def test_filter_xml_roundtrip():
    f = openmc.SensitivityFilter(['U235'], ['elastic', 'chi-delayed'],
                                 ENERGIES, filter_id=7)
    elem = f.to_xml_element()
    assert elem.get('type') == 'sensitivity'
    new_f = openmc.Filter.from_xml_element(elem)
    assert isinstance(new_f, openmc.SensitivityFilter)
    assert new_f.id == 7
    assert new_f.nuclides == f.nuclides
    assert new_f.reactions == f.reactions
    assert np.allclose(new_f.energies, f.energies)


@pytest.mark.parametrize(
    "args",
    [
        (['U235'], ['total'], ENERGIES),
        (['U235', 'U235'], ['fission'], ENERGIES),
        (['U235'], ['fission', 'fission'], ENERGIES),
        (['U235'], ['fission'], [1.0, 0.5]),
    ],
)
def test_filter_invalid(args):
    with pytest.raises(ValueError):
        openmc.SensitivityFilter(*args)


@pytest.fixture()
def model():
    openmc.reset_auto_ids()
    material = openmc.Material()
    material.add_nuclide("U235", 1.0)
    material.set_density('g/cm3', 16.0)
    sphere = openmc.Sphere(r=10.0, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)
    model = openmc.Model(geometry=openmc.Geometry([cell]))
    model.settings.particles = 1000
    model.settings.batches = 15
    model.settings.inactive = 5
    model.settings.ifp_n_generation = 5
    return model


def test_get_keff_sensitivities(run_in_tmpdir, model):
    model.add_keff_sensitivity_tallies('U235', ENERGIES)
    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        energies, coefficients = sp.get_keff_sensitivities()

    assert np.allclose(energies, ENERGIES)
    for rx in openmc.SensitivityFilter.REACTIONS + (
            'scatter', 'total', 'nu', 'chi-prompt-constrained',
            'chi-delayed-constrained'):
        assert coefficients['U235', rx].shape == (3,)

    # With a single fissionable nuclide, every fission neutron contributes
    # exactly one multiplicity event and one spectrum event, so k is exactly
    # proportional to nu and to the unconstrained fission spectrum
    nu = coefficients['U235', 'nu'].sum()
    assert nu.n == pytest.approx(1.0)
    chi = (coefficients['U235', 'chi-prompt'].sum() +
           coefficients['U235', 'chi-delayed'].sum())
    assert chi.n == pytest.approx(1.0)
    chi_constrained = coefficients['U235', 'chi-prompt-constrained'].sum()
    assert chi_constrained.n == pytest.approx(0.0, abs=1e-10)

    # Capture can only decrease k while fission increases it
    assert coefficients['U235', 'capture'].sum().n < 0.0
    assert coefficients['U235', 'fission'].sum().n > 0.0


def test_manual_tallies(run_in_tmpdir, model):
    sens_filter = openmc.SensitivityFilter('U235', ['fission', 'nu-prompt'],
                                           ENERGIES)
    sens_tally = openmc.Tally()
    sens_tally.filters = [sens_filter]
    sens_tally.scores = ['keff-sensitivity-numerator']
    denom_tally = openmc.Tally()
    denom_tally.scores = ['ifp-denominator']
    model.tallies = [sens_tally, denom_tally]

    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        _, coefficients = sp.get_keff_sensitivities()
        tally = sp.get_tally(scores=['keff-sensitivity-numerator'])
        assert tally.find_filter(openmc.SensitivityFilter) == sens_filter
    assert set(coefficients) == {('U235', 'fission'), ('U235', 'nu-prompt')}


@pytest.mark.parametrize("case", ["fixed source", "extra filter", "nuclides",
                                  "extra score"])
def test_exceptions(run_in_tmpdir, model, case):
    sens_filter = openmc.SensitivityFilter('U235', ['fission'], ENERGIES)
    tally = openmc.Tally()
    tally.filters = [sens_filter]
    tally.scores = ['keff-sensitivity-numerator']
    if case == "fixed source":
        model.settings.run_mode = "fixed source"
        model.settings.ifp_n_generation = None
    elif case == "extra filter":
        tally.filters.append(openmc.EnergyFilter(ENERGIES))
    elif case == "nuclides":
        tally.nuclides = ['U235']
    elif case == "extra score":
        tally.scores.append('flux')
    model.tallies = [tally]

    with pytest.raises(RuntimeError):
        model.run()


def test_latent_generations_xml(run_in_tmpdir):
    settings = openmc.Settings()
    settings.sensitivity_latent_generations = 3
    settings.export_to_xml()
    assert openmc.Settings.from_xml().sensitivity_latent_generations == 3
    with pytest.raises(ValueError):
        settings.sensitivity_latent_generations = -1


def test_get_kinetics_sensitivities(run_in_tmpdir, model):
    model.settings.inactive = 10
    model.settings.batches = 20
    model.settings.ifp_n_generation = 4
    model.settings.sensitivity_latent_generations = 3
    model.add_kinetics_sensitivity_tallies('U235', ENERGIES)
    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        result = sp.get_kinetics_sensitivities()

    assert np.allclose(result.energies, ENERGIES)
    assert 0.0 < result.beta_effective.n < 0.05
    assert 0.0 < result.generation_time.n < 1e-6
    for rx in ('total', 'nu', 'chi-prompt-constrained'):
        assert result.beta_coefficients['U235', rx].shape == (3,)
        assert result.generation_time_coefficients['U235', rx].shape == (3,)

    # With a single fissionable nuclide, every link of every chain contains
    # exactly one multiplicity event, so scaling nu does not change beta and
    # changes the generation time only through k
    s_beta_nu = result.beta_coefficients['U235', 'nu'].sum()
    s_gen_nu = result.generation_time_coefficients['U235', 'nu'].sum()
    assert s_beta_nu.n == pytest.approx(0.0, abs=1e-8)
    assert s_gen_nu.n == pytest.approx(-1.0)


@pytest.mark.parametrize("case", ["mixed scores", "filter on scalar",
                                  "too few inactive"])
def test_kinetics_exceptions(run_in_tmpdir, model, case):
    model.settings.inactive = 10
    model.settings.batches = 12
    model.settings.ifp_n_generation = 4
    model.settings.sensitivity_latent_generations = 3
    sens_filter = openmc.SensitivityFilter('U235', ['fission'], ENERGIES)
    tally = openmc.Tally()
    if case == "mixed scores":
        tally.filters = [sens_filter]
        tally.scores = ['beta-sensitivity-numerator', 'kinetics-population']
    elif case == "filter on scalar":
        tally.filters = [sens_filter]
        tally.scores = ['kinetics-population']
    elif case == "too few inactive":
        # 4 IFP generations + 1 + 10 latent generations > 10 inactive batches
        model.settings.sensitivity_latent_generations = 10
        tally.scores = ['kinetics-population']
    model.tallies = [tally]

    with pytest.raises(RuntimeError):
        model.run()
