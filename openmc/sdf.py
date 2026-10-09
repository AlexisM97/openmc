"""Reading and writing sensitivity data files (SDF) in the format used by the
TSUNAMI tools of SCALE."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import re
import warnings

import numpy as np
from uncertainties import ufloat, UFloat

import openmc.checkvalue as cv
from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL, zam


__all__ = ['SensitivityProfile', 'SensitivityDataFile', 'SimilarityIndices']


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
class SimilarityIndices:
    """Similarity indices of an application and an experiment.

    The indices follow the definitions of TSUNAMI-IP. The correlation
    coefficient :math:`c_k` is the correlation of the responses of the two
    systems due to nuclear data uncertainties,

    .. math::

        c_k = \\frac{S_a^T C S_e}{\\sqrt{S_a^T C S_a} \\sqrt{S_e^T C S_e}},

    where :math:`S_a` and :math:`S_e` are the sensitivities of the
    application and experiment and :math:`C` is the covariance matrix of the
    nuclear data. :math:`E` is the cosine of the angle between the
    sensitivity vectors of all nuclide-reaction pairs, excluding the total,
    scattering and capture profiles that are sums of other reactions. Its
    parts are restricted to fission (MT 18), capture (MT 102-107), and
    scattering (MT 2, 4, 16) reactions. :math:`G` measures the coverage of the
    fission, capture, and scattering sensitivities of the application by those
    of the experiment,

    .. math::

        G = 1 - \\frac{\\sum_{n,x,g} |S^a_{n,x,g} - S'^e_{n,x,g}|}
        {\\sum_{n,x,g} |S^a_{n,x,g}|},

    where :math:`S'^e` is the experiment sensitivity, limited to the
    application sensitivity when both have the same sign and set to zero when
    their signs differ.

    Uncertainties of the indices are propagated to first order from the
    statistical uncertainties of the sensitivities, assuming that the
    sensitivities of different groups and reactions are uncorrelated.

    .. versionadded:: 0.16.1

    Attributes
    ----------
    c_k : uncertainties.UFloat or None
        Correlation coefficient, or None if no covariances were given
    E : uncertainties.UFloat
        Cosine similarity of the sensitivity vectors
    E_parts : dict
        Dictionary mapping 'fission', 'capture', and 'scatter' to the
        cosine similarity of the corresponding sensitivities
    G : uncertainties.UFloat
        Coverage of the application sensitivities by the experiment
    G_parts : dict
        Dictionary mapping 'fission', 'capture', and 'scatter' to the
        coverage of the corresponding sensitivities
    application_uncertainty : float or None
        Relative uncertainty of the application response due to nuclear data
    experiment_uncertainty : float or None
        Relative uncertainty of the experiment response due to nuclear data

    """
    c_k: UFloat | None
    E: UFloat
    E_parts: dict
    G: UFloat
    G_parts: dict
    application_uncertainty: float | None = None
    experiment_uncertainty: float | None = None


# Reactions included in each part of the E and G indices
_E_PARTS = {'fission': (18,), 'capture': (102, 103, 104, 105, 106, 107),
            'scatter': (2, 4, 16)}
_G_PARTS = {'fission': 18, 'capture': 101, 'scatter': 0}


def _lethargy_transfer(energies, target):
    """Return the matrix that maps group-integrated values from one group
    structure to another, splitting each group according to lethargy"""
    u = np.log(energies)
    uc = np.log(target)
    low = np.maximum.outer(uc[:-1], u[:-1])
    high = np.minimum.outer(uc[1:], u[1:])
    return np.clip(high - low, 0.0, None) / np.diff(u)


def _cosine(blocks):
    """Return the cosine of the angle between two vectors and its uncertainty.

    Parameters
    ----------
    blocks : list of tuple
        Blocks (matrix, x, y, sx, sy) of the vectors x and y with standard
        deviations sx and sy. The inner product of each block is x^T M y, with
        the identity used if the matrix is None.

    Returns
    -------
    uncertainties.UFloat or None
        Cosine, or None if either vector has zero norm
    float, float
        Norms of x and y

    """
    def product(m, a):
        return a if m is None else m @ a

    xy = sum(x @ product(m, y) for m, x, y, _, _ in blocks)
    xx = sum(x @ product(m, x) for m, x, _, _, _ in blocks)
    yy = sum(y @ product(m, y) for m, _, y, _, _ in blocks)
    if xx <= 0.0 or yy <= 0.0:
        return None, np.sqrt(max(xx, 0.0)), np.sqrt(max(yy, 0.0))
    norm = np.sqrt(xx * yy)
    value = xy / norm
    var = 0.0
    for m, x, y, sx, sy in blocks:
        mx, my = product(m, x), product(m, y)
        var += np.sum(((my / norm - value * mx / xx) * sx)**2)
        var += np.sum(((mx / norm - value * my / yy) * sy)**2)
    return ufloat(value, np.sqrt(var)), np.sqrt(xx), np.sqrt(yy)


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

    def _integrated_profiles(self) -> dict:
        """Return the region-integrated profiles keyed by (ZA, MT)"""
        profiles = {}
        for p in self.profiles:
            if p.unit != 0 or p.region != 0:
                continue
            if (p.za, p.mt) in profiles:
                warnings.warn(f'{self.title.strip()} has several profiles '
                              f'with ZA {p.za} and MT {p.mt}; the first is '
                              'used.')
                continue
            profiles[p.za, p.mt] = p
        return profiles

    def similarity(self, experiment: SensitivityDataFile,
                   covariances: Mapping | None = None) -> SimilarityIndices:
        """Compute the similarity indices of this system and an experiment.

        This system is treated as the application. The indices follow the
        definitions of TSUNAMI-IP (see :class:`SimilarityIndices`). Profiles
        are matched by ZA and MT number, so both files should use the same
        convention for the fission spectrum profile.

        When covariances are given, :math:`c_k` and :math:`E` are computed in
        the group structure of the covariances. Sensitivities are converted to
        that structure by splitting each group in proportion to lethargy if
        the group structures differ. :math:`G` is always computed in the group
        structure of the files. Covariances between different nuclides are not
        included. Covariances are applied to the same non-redundant set of
        reactions as in :func:`openmc.data.propagate_uncertainty`; unlike
        TSUNAMI-IP, the uncertainty of nubar is therefore counted once for
        files with total, prompt, and delayed nubar profiles. The
        uncertainties of the indices are propagated to first order and are
        generally different from the approximate values of TSUNAMI-IP.

        .. versionadded:: 0.16.1

        Parameters
        ----------
        experiment : SensitivityDataFile
            Sensitivity profiles of the experiment, in the same group
            structure
        covariances : dict, optional
            Dictionary mapping nuclide names in OpenMC convention (e.g.,
            'U235') or ZA identifiers to
            :class:`openmc.data.MultigroupCovariance`. A ZA key takes
            precedence over the nuclide name, and profiles with SCALE
            identifiers above 1000000, which denote a nuclide in a particular
            material, are only matched to covariances given by ZA. Required
            for the correlation coefficient :math:`c_k`.

        Returns
        -------
        SimilarityIndices
            Similarity indices and their uncertainties

        """
        from openmc.data.covariance import _primary_mts

        cv.check_type('experiment', experiment, SensitivityDataFile)
        if len(self.energies) != len(experiment.energies) or not np.allclose(
                self.energies, experiment.energies, rtol=1e-4):
            raise ValueError('The application and experiment must have the '
                             'same group structure.')
        app = self._integrated_profiles()
        exp = experiment._integrated_profiles()
        ng = self.num_groups

        # Group structure of the covariances and conversion of sensitivities
        # to it
        transfer = None
        cov_energies = None
        if covariances:
            structures = [c.energies for c in covariances.values()]
            cov_energies = structures[0]
            if any(len(e) != len(cov_energies) or not np.allclose(
                    e, cov_energies, rtol=1e-4) for e in structures):
                raise ValueError('All covariances must have the same group '
                                 'structure.')
            if len(cov_energies) != len(self.energies) or not np.allclose(
                    cov_energies, self.energies, rtol=1e-4):
                transfer = _lethargy_transfer(self.energies, cov_energies)

        def vectors(key, convert=True):
            """Values and standard deviations of the application and
            experiment profiles, with zeros for a missing profile"""
            out = []
            for profiles in (app, exp):
                p = profiles.get(key)
                if p is None:
                    v, s = np.zeros(ng), np.zeros(ng)
                else:
                    v = np.asarray(p.values, dtype=float)
                    s = np.asarray(p.std_dev, dtype=float)
                if convert and transfer is not None:
                    v, s = transfer @ v, np.sqrt(transfer**2 @ s**2)
                out += [v, s]
            return out

        # Correlation coefficient
        c_k = unc_app = unc_exp = None
        if covariances:
            nuclides = {}
            for p in list(app.values()) + list(exp.values()):
                nuclides.setdefault(p.za, p.openmc_nuclide)
            blocks = []
            for za, name in sorted(nuclides.items()):
                # SCALE identifiers above 1000000 denote a nuclide in a
                # particular material and only match covariances given by ZA
                cov = covariances.get(za)
                if cov is None and za < 1000000:
                    cov = covariances.get(name)
                if cov is None:
                    continue
                mts = _primary_mts(cov)
                n = cov.num_groups
                matrix = np.zeros((n * len(mts), n * len(mts)))
                for i, mt1 in enumerate(mts):
                    for j, mt2 in enumerate(mts):
                        m = cov.get(mt1, mt2)
                        if m is not None:
                            matrix[i*n:(i + 1)*n, j*n:(j + 1)*n] = m
                x, sx, y, sy = (np.concatenate(v) for v in zip(
                    *(vectors((za, mt)) for mt in mts)))
                blocks.append((matrix, x, y, sx, sy))
            c_k, unc_app, unc_exp = _cosine(blocks)

        # Cosine similarity of all sensitivities except sums of reactions
        keys = sorted(k for k in set(app) | set(exp) if k[1] not in (0, 1, 101))
        blocks = {}
        for k in keys:
            x, sx, y, sy = vectors(k)
            blocks[k] = (None, x, y, sx, sy)
        E = _cosine(list(blocks.values()))[0]
        E_parts = {}
        for part, mts in _E_PARTS.items():
            value = _cosine([b for k, b in blocks.items() if k[1] in mts])[0]
            if value is not None:
                E_parts[part] = value

        # Coverage of the application sensitivities
        def coverage(mts):
            total = 0.0
            diff = 0.0
            terms = []
            for (za, mt), p in app.items():
                if mt not in mts:
                    continue
                s_a, sd_a, s_e, sd_e = vectors((za, mt), convert=False)
                sign_a = np.copysign(1.0, s_a)
                sign_e = np.copysign(1.0, s_e)
                included = p.integral != 0.0
                # Experiment sensitivities of the same sign cover the
                # application up to its magnitude; those of opposite sign do
                # not cover it
                same = (sign_a == sign_e) & (np.abs(s_a) >= np.abs(s_e)) & \
                    included
                opposite = (sign_a != sign_e) & included
                d = np.where(same, s_a - s_e, np.where(opposite, s_a, 0.0))
                total += np.abs(s_a).sum()
                diff += np.abs(d).sum()
                terms.append((s_a, sd_a, sd_e, d, same))
            if total == 0.0:
                return None
            value = 1.0 - diff / total
            var = 0.0
            for s_a, sd_a, sd_e, d, same in terms:
                grad_a = -np.sign(d) / total + diff * np.sign(s_a) / total**2
                grad_e = np.where(same, np.sign(d), 0.0) / total
                var += np.sum((grad_a * sd_a)**2) + np.sum((grad_e * sd_e)**2)
            return ufloat(value, np.sqrt(var))

        G = coverage(set(_G_PARTS.values()))
        G_parts = {}
        for part, mt in _G_PARTS.items():
            value = coverage({mt})
            if value is not None:
                G_parts[part] = value

        return SimilarityIndices(c_k, E, E_parts, G, G_parts, unc_app,
                                 unc_exp)

    @classmethod
    def from_coefficients(cls, coefficients: Mapping, energies, keff=None,
                          title: str = 'OpenMC sensitivity data',
                          chi: str = 'unconstrained'):
        """Create a sensitivity data file from OpenMC sensitivity coefficients.

        Profiles are written for each nuclide for the total, scattering,
        elastic, and capture cross sections and, when the corresponding
        coefficients are available and nonzero, for individual reactions, the
        fission cross section, the total fission neutron multiplicity (nubar),
        and the fission spectrum (chi). Tallying the 'elastic',
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
        chi : {'constrained', 'unconstrained'}
            Whether the fission spectrum profile preserves the normalization
            of the spectrum, in which case it integrates to zero, or is the
            unconstrained sensitivity, which integrates to the sensitivity to
            nubar. Both conventions are found in sensitivity data files:
            MCNP and Whisper profiles are constrained, while TSUNAMI-IP
            treats profiles as unconstrained by default. Profiles compared
            with each other, for instance with :meth:`similarity`, should use
            the same convention.

        Returns
        -------
        SensitivityDataFile
            Sensitivity data file

        """
        cv.check_value('chi', chi, ('constrained', 'unconstrained'))
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
                if rx == 'chi' and chi == 'unconstrained':
                    keys = ['chi-prompt', 'chi-delayed']
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
            nuclide, mt = fields[0], int(fields[-1])
            # A ZA identifier that fills its field runs into the reaction
            # name (e.g., 'scatter3001001')
            m = re.fullmatch(r'(.*?)(-?\d+)', fields[-2])
            za = int(m.group(2))
            reaction = ' '.join(fields[1:-2] + [m.group(1)]).strip()
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
                 f'{self.keff:14.6E} +/- {self.keff_std_dev:14.6E} k-eff '
                 'from the forward case',
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
