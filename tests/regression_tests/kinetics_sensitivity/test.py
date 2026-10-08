"""Test the calculation of sensitivity coefficients of kinetics parameters
using the collision history-based method."""

import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture()
def kinetics_model():
    model = openmc.Model()

    # Material
    material = openmc.Material(name="core")
    material.add_nuclide("U235", 0.9)
    material.add_nuclide("U238", 0.1)
    material.set_density('g/cm3', 18.0)

    # Geometry
    radius = 9.0
    sphere = openmc.Sphere(r=radius, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)
    model.geometry = openmc.Geometry([cell])

    # Settings
    model.settings.particles = 1000
    model.settings.batches = 20
    model.settings.inactive = 10
    model.settings.ifp_n_generation = 4
    model.settings.sensitivity_latent_generations = 3

    space = openmc.stats.Box(*cell.bounding_box)
    model.settings.source = openmc.IndependentSource(
        space=space, constraints={'fissionable': True})

    # Tally kinetics parameter sensitivity coefficients
    model.add_kinetics_sensitivity_tallies(
        ['U235', 'U238'], [1e-5, 1e5, 1e6, 2e7],
        ['elastic', 'inelastic', 'capture', 'fission', 'nu-prompt',
         'nu-delayed'])

    return model


def test_kinetics_sensitivity(kinetics_model):
    harness = PyAPITestHarness("statepoint.20.h5", model=kinetics_model)
    harness.main()
