"""Reading and writing sensitivity data files (SDF) in the format used by the
TSUNAMI tools of SCALE."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import re

import numpy as np

import openmc.checkvalue as cv
from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL, zam


__all__ = ['SensitivityProfile', 'SensitivityDataFile']


# Profiles written for each nuclide: SDF reaction name, MT number, and the
# names of the OpenMC sensitivity coefficients that are summed to form it
_SDF_REACTIONS = [
    ('total', 1, ['total']),
    ('scatter', 0, ['scatter']),
    ('elastic', 2, ['elastic', '(n,elastic)']),
    ("n,n'", 4, ['(n,level)']),
    ('n,2n', 16, ['(n,2n)']),
    ('n,3n', 17, ['(n,3n)']),
    ('fission', 18, ['fission']),
    ('capture', 101, ['capture']),
    ('n,gamma', 102, ['(n,gamma)']),
    ('n,p', 103, ['(n,p)']),
    ('n,d', 104, ['(n,d)']),
    ('n,t', 105, ['(n,t)']),
    ('n,he-3', 106, ['(n,3He)']),
    ('n,alpha', 107, ['(n,a)']),
    ('nubar', 452, ['nu']),
    ('chi', 1018, ['chi-prompt-constrained', 'chi-delayed-constrained']),
]

# Profiles always written when their data are available, even if zero
_REQUIRED = {'total', 'scatter', 'elastic', 'capture'}


def _sdf_nuclide_name(nuclide: str) -> tuple[str, int]:
    """Convert an OpenMC nuclide name to an SDF name and ZA identifier"""
    z, a, m = zam(nuclide)
    symbol = ATOMIC_SYMBOL[z].lower()
    if a == 0:
        return symbol, 1000 * z
    name = f'{symbol}-{a}' + ('m' if m > 0 else '')
    return name, 1000 * z + a + (300 + 100 * m if m > 0 else 0)


def _openmc_nuclide_name(name: str) -> str:
    """Convert an SDF nuclide name to an OpenMC nuclide name"""
    m = re.fullmatch(r'([a-z]+)-?(\d*)(m?)', name.lower())
    if m is None or m.group(1).capitalize() not in ATOMIC_NUMBER:
        return name
    symbol = m.group(1).capitalize()
    if not m.group(2):
        return f'{symbol}0'
    return f'{symbol}{m.group(2)}' + ('_m1' if m.group(3) else '')


@dataclass
class SensitivityProfile:
    """Sensitivity coefficients of a response to a nuclide-reaction pair.

    .. versionadded:: 0.16.1

    Attributes
    ----------
    nuclide : str
        Name of the nuclide in SDF convention (e.g., 'u-235')
    reaction : str
        Name of the reaction in SDF convention (e.g., 'fission' or 'n,gamma')
    za : int
        ZA identifier of the nuclide
    mt : int
        Reaction identifier (MT number, 0 for scatter, 452 for nubar, and 1018
        for the fission spectrum)
    values : numpy.ndarray
        Sensitivity coefficients in each group, ordered by increasing energy
    std_dev : numpy.ndarray
        Standard deviations of the sensitivity coefficients
    unit : int
        Unit number, 0 for a sensitivity that is not specific to a unit
    region : int
        Region number, 0 for a region-integrated sensitivity

    """
    nuclide: str
    reaction: str
    za: int
    mt: int
    values: np.ndarray
    std_dev: np.ndarray
    unit: int = 0
    region: int = 0

    @property
    def openmc_nuclide(self) -> str:
        """Name of the nuclide in OpenMC convention"""
        return _openmc_nuclide_name(self.nuclide)

    @property
    def integral(self) -> float:
        return float(np.sum(self.values))

    @property
    def integral_std_dev(self) -> float:
        """Standard deviation of the energy-integrated sensitivity, assuming
        uncorrelated groups"""
        return float(np.sqrt(np.sum(self.std_dev**2)))


@dataclass
class SensitivityDataFile:
    """Collection of sensitivity profiles in a sensitivity data file (SDF).

    Sensitivity data files are produced by the TSUNAMI tools of SCALE and are
    used by TSUNAMI-IP and other tools for similarity assessment and
    uncertainty quantification. Profiles contain sensitivity coefficients
    integrated over each energy group (not per unit lethargy).

    .. versionadded:: 0.16.1

    Attributes
    ----------
    title : str
        Title of the file
    energies : numpy.ndarray
        Energy group boundaries in [eV] in increasing order
    keff : float
        Multiplication factor of the system
    keff_std_dev : float
        Standard deviation of the multiplication factor
    profiles : list of SensitivityProfile
        Sensitivity profiles

    """
    title: str
    energies: np.ndarray
    keff: float = 0.0
    keff_std_dev: float = 0.0
    profiles: list = field(default_factory=list)

    @property
    def num_groups(self) -> int:
        return len(self.energies) - 1

    def get(self, nuclide: str, reaction: str) -> SensitivityProfile | None:
        """Return the region-integrated profile of a nuclide-reaction pair.

        Parameters
        ----------
        nuclide : str
            Name of the nuclide in SDF (e.g., 'u-235') or OpenMC (e.g.,
            'U235') convention
        reaction : str
            Name of the reaction in SDF convention (e.g., 'fission')

        Returns
        -------
        SensitivityProfile or None
            Profile, or None if the file does not contain it

        """
        for p in self.profiles:
            if p.region == 0 and p.unit == 0 and p.reaction == reaction and \
                    nuclide in (p.nuclide, p.openmc_nuclide):
                return p
        return None

    @classmethod
    def from_coefficients(cls, coefficients: Mapping, energies, keff=None,
                          title: str = 'OpenMC sensitivity data'):
        """Create a sensitivity data file from OpenMC sensitivity coefficients.

        Profiles are written for each nuclide for the total, scattering,
        elastic, and capture cross sections and, when the corresponding
        coefficients are available and nonzero, for individual reactions, the
        fission cross section, the total fission neutron multiplicity (nubar),
        and the constrained fission spectrum (chi). Tallying the 'elastic',
        'inelastic', 'capture', and 'fission' data is required for the total,
        scattering, and capture profiles.

        Parameters
        ----------
        coefficients : dict
            Dictionary mapping (nuclide, reaction) to sensitivity coefficients
            in each group, as returned by
            :meth:`openmc.StatePoint.get_keff_sensitivities`
        energies : Iterable of float
            Energy group boundaries in [eV] in increasing order
        keff : float or uncertainties.UFloat, optional
            Multiplication factor of the system
        title : str, optional
            Title of the file

        Returns
        -------
        SensitivityDataFile
            Sensitivity data file

        """
        energies = np.asarray(energies, dtype=float)
        ng = len(energies) - 1

        def split(values):
            values = np.asarray(values)
            if values.dtype == object:
                return (np.array([getattr(v, 'n', v) for v in values], float),
                        np.array([getattr(v, 's', 0.0) for v in values],
                                 float))
            return values.astype(float), np.zeros(ng)

        keff_n = getattr(keff, 'n', keff) or 0.0
        keff_s = getattr(keff, 's', 0.0)
        sdf = cls(title, energies, float(keff_n), float(keff_s))

        nuclides = sorted({nuc for nuc, _ in coefficients},
                          key=lambda n: (zam(n), n))
        for nuc in nuclides:
            name, za = _sdf_nuclide_name(nuc)
            for rx, mt, keys in _SDF_REACTIONS:
                present = [k for k in keys if (nuc, k) in coefficients]
                if not present:
                    continue
                # Alternative names refer to the same data; only the chi
                # profile sums the prompt and delayed contributions
                if rx != 'chi':
                    present = present[:1]
                parts = [split(coefficients[nuc, k]) for k in present]
                values = sum(p[0] for p in parts)
                std = np.sqrt(sum(p[1]**2 for p in parts))
                if rx not in _REQUIRED and not np.any(values):
                    continue
                sdf.profiles.append(SensitivityProfile(
                    name, rx, za, mt, values, std))
        return sdf

    @classmethod
    def from_file(cls, filename: str | os.PathLike):
        """Read a sensitivity data file.

        Parameters
        ----------
        filename : str or os.PathLike
            Path to the file

        Returns
        -------
        SensitivityDataFile
            Sensitivity data file

        """
        with open(filename) as fh:
            lines = fh.read().splitlines()

        title = lines[0].strip()
        ng = int(lines[1].split()[0])
        n_profiles = int(lines[2].split()[0])
        tokens = lines[3].split()
        keff = float(tokens[0])
        keff_std = float(tokens[2]) if len(tokens) > 2 and \
            tokens[1] == '+/-' else 0.0

        # Read values in fixed-width fields of 14 characters, which may run
        # together when negative
        def numbers(line):
            line = line.rstrip()
            if 'E' in line or 'e' in line:
                return [float(line[i:i + 14]) for i in range(0, len(line), 14)
                        if line[i:i + 14].strip()]
            return [float(x) for x in line.split()]

        pos = 4
        while not lines[pos].strip().lower().startswith('energy boundaries'):
            pos += 1
        pos += 1
        bounds = []
        while len(bounds) < ng + 1:
            bounds += numbers(lines[pos])
            pos += 1
        energies = np.array(bounds[::-1])
        sdf = cls(title, energies, keff, keff_std)

        for _ in range(n_profiles):
            while not lines[pos].strip():
                pos += 1
            fields = lines[pos].split()
            nuclide, za, mt = fields[0], int(fields[-2]), int(fields[-1])
            reaction = ' '.join(fields[1:-2])
            unit, region = (int(x) for x in lines[pos + 1].split()[:2])
            pos += 4
            data = []
            while len(data) < 2 * ng:
                data += numbers(lines[pos])
                pos += 1
            values = np.array(data[:ng])[::-1]
            std = np.array(data[ng:2 * ng])[::-1]
            sdf.profiles.append(SensitivityProfile(
                nuclide, reaction, za, mt, values, std, unit, region))
        return sdf

    def export_to_file(self, filename: str | os.PathLike):
        """Write the sensitivity data file in the format of TSUNAMI.

        Parameters
        ----------
        filename : str or os.PathLike
            Path of the file to write

        """
        cv.check_length('energies', self.energies, 2)

        def block(values):
            out = []
            for i in range(0, len(values), 5):
                out.append(''.join(f'{v:14.6E}' for v in values[i:i + 5]))
            return out

        lines = [f'{self.title[:80]:80s}',
                 f'{self.num_groups:10d}   number of neutron groups',
                 f'{len(self.profiles):10d}   number of sensitivity profiles'
                 f'{len(self.profiles):12d} are region integrated',
                 f'{self.keff:10.6f} +/- {self.keff_std_dev:10.6f} k-eff from '
                 'the forward case',
                 'energy boundaries:']
        lines += block(self.energies[::-1])

        for p in self.profiles:
            values = np.asarray(p.values, dtype=float)
            std = np.asarray(p.std_dev, dtype=float)
            total = values.sum()
            # Sum of the values whose sign is opposite to that of the
            # integral and its standard deviation
            opposite = values * np.sign(total) < 0.0
            lines.append(f'{p.nuclide:<13s}{p.reaction:<17s}{p.za:5d}'
                         f'{p.mt:7d}')
            lines.append(f'{p.unit:7d}{p.region:7d}')
            lines.append(f'{0.0:14.6E}{0.0:14.6E}{0:7d}{0:7d}')
            lines.append(''.join(f'{v:14.6E}' for v in (
                total, np.sqrt(np.sum(std**2)), np.abs(values).sum(),
                values[opposite].sum(), np.sqrt(np.sum(std[opposite]**2)))))
            lines += block(values[::-1])
            lines += block(std[::-1])

        with open(filename, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
