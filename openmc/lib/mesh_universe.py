"""Element data of mesh universes: temperatures and density multipliers to
set from a coupled physics code, element volumes and blocks"""
from ctypes import c_int, c_int32, c_double, POINTER

import numpy as np

from . import _dll
from .error import _error_handler

__all__ = ['MeshUniverse']

_dll.openmc_mesh_universe_get_n_elements.argtypes = [c_int32, POINTER(c_int32)]
_dll.openmc_mesh_universe_get_blocks.argtypes = [
    c_int32, POINTER(c_int32), c_int32]
for _name in ('volumes', 'temperatures', 'density_mults'):
    getattr(_dll, f'openmc_mesh_universe_get_{_name}').argtypes = [
        c_int32, POINTER(c_double), c_int32]
for _name in ('temperatures', 'density_mults'):
    getattr(_dll, f'openmc_mesh_universe_set_{_name}').argtypes = [
        c_int32, POINTER(c_double), c_int32]
for _name in ('get_n_elements', 'get_blocks', 'get_volumes',
              'get_temperatures', 'set_temperatures', 'get_density_mults',
              'set_density_mults'):
    _f = getattr(_dll, f'openmc_mesh_universe_{_name}')
    _f.restype = c_int
    _f.errcheck = _error_handler


class MeshUniverse:
    """Mesh universe of the loaded model, by ID: element temperatures and
    density multipliers (in the order of the elements of the mesh file)

    Parameters
    ----------
    universe_id : int
        ID of the mesh universe

    Attributes
    ----------
    n_elements : int
        Number of elements
    blocks : numpy.ndarray
        Element block ID of each element
    volumes : numpy.ndarray
        Volume of each element [cm^3]
    temperatures : numpy.ndarray
        Temperature of each element [K]; elements without a temperature of
        their own take that of their cell
    density_multipliers : numpy.ndarray
        Density multiplier of each element

    """

    def __init__(self, universe_id):
        self.id = universe_id
        n = c_int32()
        _dll.openmc_mesh_universe_get_n_elements(universe_id, n)
        self.n_elements = n.value

    def _get(self, name, dtype=np.float64, ctype=c_double):
        x = np.empty(self.n_elements, dtype=dtype)
        getattr(_dll, f'openmc_mesh_universe_get_{name}')(
            self.id, x.ctypes.data_as(POINTER(ctype)), self.n_elements)
        return x

    def _set(self, name, values):
        x = np.ascontiguousarray(values, dtype=np.float64)
        if x.shape != (self.n_elements,):
            raise ValueError(f'Expected {self.n_elements} values, got '
                             f'{x.size}.')
        getattr(_dll, f'openmc_mesh_universe_set_{name}')(
            self.id, x.ctypes.data_as(POINTER(c_double)), self.n_elements)

    @property
    def blocks(self):
        return self._get('blocks', np.int32, c_int32)

    @property
    def volumes(self):
        return self._get('volumes')

    @property
    def temperatures(self):
        return self._get('temperatures')

    @temperatures.setter
    def temperatures(self, T):
        self._set('temperatures', T)

    @property
    def density_multipliers(self):
        return self._get('density_mults')

    @density_multipliers.setter
    def density_multipliers(self, mult):
        self._set('density_mults', mult)
