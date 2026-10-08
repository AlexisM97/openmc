"""Test the calculation of k-effective sensitivity coefficients using the
collision history-based method."""

import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture()
def sensitivity_model():
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
    model.settings.inactive = 5
    model.settings.ifp_n_generation = 5

    space = openmc.stats.Box(*cell.bounding_box)
    model.settings.source = openmc.IndependentSource(
        space=space, constraints={'fissionable': True})

    # Tally sensitivity coefficients
    model.add_keff_sensitivity_tallies(
        ['U235', 'U238'], [1e-5, 1e5, 1e6, 2e7])

    return model


def test_keff_sensitivity(sensitivity_model):
    harness = PyAPITestHarness("statepoint.20.h5", model=sensitivity_model)
    harness.main()
