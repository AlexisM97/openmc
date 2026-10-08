"""Multigroup covariances of nuclear data and propagation of nuclear data
uncertainties using sensitivity coefficients."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
import os
from pathlib import Path
import tempfile
import warnings

import h5py
import numpy as np

import openmc.checkvalue as cv
from . import endf
from .njoy import run
from .reaction import REACTION_MT, REACTION_NAME

__all__ = ['MultigroupCovariance', 'Uncertainty', 'UncertaintyContribution',
           'propagate_uncertainty']

# Covariance key used for fission spectra (MF35), following the convention
# of SCALE covariance libraries
CHI_MT = 1018

# Lumped reaction covariances (MT 851-870) defined by NJOY
_LUMPED_MTS = range(851, 871)


def _label(mt: int) -> str:
    """Return a readable label for a covariance key"""
    special = {452: 'nubar', 455: 'nubar-delayed', 456: 'nubar-prompt',
               CHI_MT: 'chi'}
    if mt in special:
        return special[mt]
    if mt in _LUMPED_MTS:
        return f'lumped MT={mt}'
    return REACTION_NAME.get(mt, f'MT={mt}')


class MultigroupCovariance:
    """Relative multigroup covariances of the nuclear data of a nuclide.

    Covariances are stored as relative covariance matrices between pairs of
    reactions in an energy group structure. Reactions are identified by their
    MT number, with total, prompt and delayed fission neutron multiplicities
    identified by MT 452, 456 and 455, and the fission spectrum by MT 1018.
    Groups are ordered by increasing energy.

    .. versionadded:: 0.16.1

    Parameters
    ----------
    nuclide : str
        Name of the nuclide, e.g. 'U235'
    energies : Iterable of float
        Energy group boundaries in [eV] in increasing order
    covariances : dict
        Dictionary mapping a pair of MT numbers to a relative covariance matrix
    cross_sections : dict, optional
        Dictionary mapping MT numbers to multigroup cross sections in [b] (or
        multiplicities and spectra) used as weights when processing
    lumped : dict, optional
        Dictionary mapping the MT number of a reaction to the MT number of the
        lumped reaction (MT 851-870) whose covariances include it
    temperature : float, optional
        Temperature in [K] at which the data were processed

    Attributes
    ----------
    nuclide : str
        Name of the nuclide
    energies : numpy.ndarray
        Energy group boundaries in [eV]
    covariances : dict
        Dictionary mapping a pair of MT numbers to a relative covariance matrix
    cross_sections : dict
        Dictionary mapping MT numbers to multigroup values
    lumped : dict
        Dictionary mapping MT numbers to lumped reaction MT numbers
    temperature : float
        Temperature in [K] at which the data were processed
    mts : list of int
        MT numbers with variances

    """

    def __init__(self, nuclide, energies, covariances, cross_sections=None,
                 lumped=None, temperature=None):
        self.nuclide = nuclide
        self.energies = np.asarray(energies, dtype=float)
        self.covariances = {(int(a), int(b)): np.asarray(m, dtype=float)
                            for (a, b), m in covariances.items()}
        self.cross_sections = dict(cross_sections or {})
        self.lumped = dict(lumped or {})
        self.temperature = temperature

    def __repr__(self):
        return (f'<MultigroupCovariance: {self.nuclide}, '
                f'{self.num_groups} groups, MTs {self.mts}>')

    @property
    def num_groups(self):
        return len(self.energies) - 1

    @property
    def mts(self):
        return sorted({a for a, b in self.covariances if a == b})

    def get(self, mt1: int, mt2: int) -> np.ndarray | None:
        """Return the relative covariance matrix between two reactions.

        Parameters
        ----------
        mt1, mt2 : int
            MT numbers of the reactions

        Returns
        -------
        numpy.ndarray or None
            Relative covariance matrix whose rows correspond to the groups of
            mt1 and columns to the groups of mt2, or None if there is no
            covariance between the reactions

        """
        if (mt1, mt2) in self.covariances:
            return self.covariances[mt1, mt2]
        if (mt2, mt1) in self.covariances:
            return self.covariances[mt2, mt1].T
        return None

    def std_dev(self, mt: int) -> np.ndarray:
        """Return the relative standard deviation of a reaction in each group.

        Parameters
        ----------
        mt : int
            MT number of the reaction

        Returns
        -------
        numpy.ndarray
            Relative standard deviation in each group

        """
        return np.sqrt(np.maximum(np.diag(self.get(mt, mt)), 0.0))

    def correlation(self, mt1: int, mt2: int) -> np.ndarray:
        """Return the correlation matrix between two reactions.

        Parameters
        ----------
        mt1, mt2 : int
            MT numbers of the reactions

        Returns
        -------
        numpy.ndarray
            Correlation matrix

        """
        cov = self.get(mt1, mt2)
        s1, s2 = self.std_dev(mt1), self.std_dev(mt2)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / np.outer(s1, s2)
        return np.nan_to_num(corr)

    def export_to_hdf5(self, path, mode='a'):
        """Export covariance data to an HDF5 file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to write HDF5 file to
        mode : {'r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file

        """
        with h5py.File(path, mode) as f:
            if self.nuclide in f:
                del f[self.nuclide]
            g = f.create_group(self.nuclide)
            g.attrs['filetype'] = np.bytes_('multigroup_covariance')
            if self.temperature is not None:
                g.attrs['temperature'] = self.temperature
            g.create_dataset('energies', data=self.energies)
            cov = g.create_group('covariances')
            for (a, b), matrix in self.covariances.items():
                cov.create_dataset(f'{a}_{b}', data=matrix)
            xs = g.create_group('cross_sections')
            for mt, values in self.cross_sections.items():
                xs.create_dataset(str(mt), data=values)
            if self.lumped:
                g.create_dataset('lumped', data=np.array(
                    list(self.lumped.items()), dtype=int))

    @classmethod
    def from_hdf5(cls, path, nuclide: str | None = None):
        """Load covariance data from an HDF5 file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to an HDF5 file written by :meth:`export_to_hdf5`
        nuclide : str, optional
            Name of the nuclide to load. Required if the file contains more
            than one nuclide.

        Returns
        -------
        MultigroupCovariance
            Covariance data of the nuclide

        """
        with h5py.File(path, 'r') as f:
            if nuclide is None:
                if len(f) != 1:
                    raise ValueError('The file contains several nuclides; '
                                     'specify which one to load.')
                nuclide = list(f)[0]
            g = f[nuclide]
            covariances = {}
            for name, dset in g['covariances'].items():
                a, b = map(int, name.split('_'))
                covariances[a, b] = dset[()]
            cross_sections = {int(name): dset[()]
                              for name, dset in g['cross_sections'].items()}
            lumped = {}
            if 'lumped' in g:
                lumped = {int(a): int(b) for a, b in g['lumped'][()]}
            temperature = g.attrs.get('temperature')
            return cls(nuclide, g['energies'][()], covariances,
                       cross_sections, lumped, temperature)

    @classmethod
    def from_errorr(cls, filenames, nuclide: str):
        """Read covariances from ENDF-formatted output of NJOY's ERRORR module.

        The files must have been produced with relative covariances
        (``irelco=1``). Fission spectrum covariances (MF35) are stored with
        MT 1018 and multiplicity covariances (MF31), which ERRORR writes in
        MF33, with MT 452, 455 or 456.

        Parameters
        ----------
        filenames : str, os.PathLike, or iterable thereof
            ERRORR output files
        nuclide : str
            Name of the nuclide

        Returns
        -------
        MultigroupCovariance
            Covariance data of the nuclide

        """
        if isinstance(filenames, (str, os.PathLike)):
            filenames = [filenames]

        energies = None
        temperature = None
        covariances = {}
        cross_sections = {}
        lumped = {}
        for filename in filenames:
            is_chi = _is_chi(filename)
            for mf, mt, text in _errorr_sections(filename):
                fh = StringIO(text)
                if mf == 1 and mt == 451:
                    endf.get_head_record(fh)
                    items, values = endf.get_list_record(fh)
                    temperature = items[0]
                    energies = np.asarray(values)
                elif mf == 3:
                    _, values = endf.get_list_record(fh)
                    key = CHI_MT if mt == 18 and is_chi else mt
                    cross_sections[key] = np.asarray(values)
                elif mf in (33, 35):
                    key = CHI_MT if mf == 35 else mt
                    _, _, _, mt_lump, _, n_sub = endf.get_head_record(fh)
                    if mt_lump in _LUMPED_MTS and mt not in _LUMPED_MTS:
                        lumped[mt] = mt_lump
                    for _ in range(n_sub):
                        _, _, mat1, mt1, _, ng = endf.get_cont_record(fh)
                        matrix = np.zeros((ng, ng))
                        while True:
                            items, values = endf.get_list_record(fh)
                            ig2lo, ig = items[3], items[5]
                            matrix[ig - 1, ig2lo - 1:ig2lo - 1 + len(values)] \
                                = values
                            if ig == ng:
                                break
                        if mat1 != 0:
                            warnings.warn(
                                f'Covariances between {nuclide} MT={mt} and '
                                f'another material (MAT={mat1}) are ignored.')
                            continue
                        key1 = CHI_MT if mf == 35 else mt1
                        covariances[key, key1] = matrix

        if energies is None:
            raise ValueError('Could not find the group structure in the '
                             'ERRORR output.')
        return cls(nuclide, energies, covariances, cross_sections, lumped,
                   temperature)

    @classmethod
    def from_endf(cls, filename, energies, temperature: float = 293.6,
                  covariances: Sequence[int] = (31, 33, 35),
                  fission_energy: float = 2.0e6, weight_function: int = 6,
                  error: float = 0.001, njoy_exec: str = 'njoy',
                  output_dir=None, verbose: bool = False):
        """Process covariances from an ENDF-6 evaluation with NJOY.

        The evaluation is reconstructed with RECONR, Doppler broadened with
        BROADR and group-averaged with GROUPR before relative covariances are
        computed with ERRORR for cross sections (MF33), fission neutron
        multiplicities (MF31) and fission spectra (MF35). Files that are not
        present in the evaluation are skipped.

        Parameters
        ----------
        filename : str or os.PathLike
            Path to the ENDF-6 evaluation
        energies : Iterable of float or str
            Energy group boundaries in [eV] in increasing order, or the name of
            a group structure in :data:`openmc.mgxs.GROUP_STRUCTURES`. This
            should match the group structure of the sensitivity coefficients.
        temperature : float, optional
            Temperature in [K]
        covariances : iterable of int, optional
            ENDF files of covariances to process (31, 33, and/or 35)
        fission_energy : float, optional
            Incident neutron energy in [eV] used to select the fission spectrum
            covariances when they depend on incident energy. Evaluations
            such as ENDF/B-VIII.0 U235 give separate covariances for several
            incident energy ranges, so this should be representative of the
            energies of neutrons causing fission in the system of interest.
        weight_function : int, optional
            NJOY weight function option (``iwt``) for GROUPR and ERRORR. The
            default, 6, uses a thermal Maxwellian, a 1/E slowing-down
            spectrum, and a fission spectrum.
        error : float, optional
            Fractional error tolerance for RECONR and BROADR
        njoy_exec : str, optional
            Path to the NJOY executable
        output_dir : str or os.PathLike, optional
            Directory in which to keep the NJOY input and ERRORR output files.
            If not given, they are discarded.
        verbose : bool, optional
            Whether to display NJOY output

        Returns
        -------
        MultigroupCovariance
            Covariance data of the nuclide

        """
        if isinstance(energies, str):
            import openmc.mgxs
            energies = openmc.mgxs.GROUP_STRUCTURES[energies.upper()]
        energies = np.asarray(energies, dtype=float)
        cv.check_length('energies', energies, 2)
        if np.any(np.diff(energies) <= 0.0):
            raise ValueError('Energy group boundaries must be increasing.')

        ev = endf.Evaluation(filename)
        mat = ev.material
        nuclide = ev.gnds_name
        present = {mf for mf, mt in ev.section}
        files = [mf for mf in covariances if mf in present]
        if 33 not in present and 33 in covariances:
            warnings.warn(f'{nuclide} has no cross section covariances.')
        if not files:
            raise ValueError(f'{nuclide} has none of the requested '
                             'covariance files.')

        ng = len(energies) - 1
        groups = f'{ng} /\n' + '\n'.join(
            ' '.join(f'{e:.8e}' for e in energies[i:i + 5])
            for i in range(0, len(energies), 5)) + ' /\n'
        # Fission neutron multiplicities and spectra are needed by ERRORR for
        # MF31 and MF35 covariances
        fission = ''
        if (1, 452) in ev.section:
            fission += '3 452 /\n'
            for mt in (455, 456):
                if (1, mt) in ev.section:
                    fission += f'3 {mt} /\n'
        if 35 in files:
            fission += '5 18 /\n'
        commands = _TEMPLATE_PREPROCESS.format(
            mat=mat, nuclide=nuclide, error=error, temperature=temperature,
            iwt=weight_function, groups=groups, fission=fission)
        tapes = {33: 30, 31: 31, 35: 35}
        for mf in files:
            npend = -23 if mf == 33 else 0
            extra = f' {fission_energy:.6e}' if mf == 35 else ''
            commands += _TEMPLATE_ERRORR.format(
                npend=npend, nout=tapes[mf], mat=mat, iwt=weight_function,
                temperature=temperature, mfcov=mf, extra=extra,
                groups=groups)
        commands += 'stop\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(output_dir) if output_dir is not None else Path(tmpdir)
            outdir.mkdir(parents=True, exist_ok=True)
            tapeout = {tapes[mf]: outdir / f'{nuclide}.errorr{mf}'
                       for mf in files}
            run(commands, {20: filename}, tapeout,
                input_filename=outdir / f'{nuclide}.njoy.inp',
                stdout=verbose, njoy_exec=njoy_exec)
            missing = [str(p) for p in tapeout.values() if not p.is_file()]
            if missing:
                raise RuntimeError(f'NJOY did not produce {missing}.')
            data = cls.from_errorr(list(tapeout.values()), nuclide)

        data.temperature = temperature
        if not np.allclose(data.energies, energies, rtol=1e-4):
            raise RuntimeError('The group structure of the ERRORR output does '
                               'not match the requested one.')
        data.energies = energies
        return data


_TEMPLATE_PREPROCESS = """moder
20 -21 /
reconr
-21 -22 /
'{nuclide} covariances'/
{mat} 0 0 /
{error} /
0 /
broadr
-21 -22 -23 /
{mat} 1 0 0 0. /
{error} /
{temperature} /
0 /
groupr
-21 -23 0 -24 /
{mat} 1 0 {iwt} 0 1 1 0 /
'{nuclide}' /
{temperature} /
1.0e10 /
{groups}3 /
{fission}0 /
0 /
"""

_TEMPLATE_ERRORR = """errorr
-21 {npend} -24 {nout} 0 0 /
{mat} 1 {iwt} 0 1 /
0 {temperature} /
0 {mfcov} 1 1 -1{extra} /
{groups}"""


def _errorr_sections(filename):
    """Yield (MF, MT, text) for each section of an ENDF-formatted file"""
    sections = []
    current = None
    with open(filename) as fh:
        for line in fh:
            if len(line) < 75:
                continue
            try:
                mf = int(line[70:72])
                mt = int(line[72:75])
            except ValueError:
                continue
            if mf == 0 or mt == 0:
                current = None
                continue
            if current is None or current[:2] != (mf, mt):
                current = (mf, mt, [])
                sections.append(current)
            current[2].append(line)
    for mf, mt, lines in sections:
        yield mf, mt, ''.join(lines)


def _is_chi(filename):
    """Whether an ERRORR output file contains fission spectrum covariances"""
    with open(filename) as fh:
        return any(line[70:72] == '35' for line in fh)


# Covariance keys and the names of the corresponding sensitivity coefficients,
# in order of preference
def _sensitivity_names(mt):
    if mt == 2:
        return ['(n,elastic)', 'elastic']
    if mt == 18:
        return ['fission', '(n,fission)']
    if mt == 452:
        return ['nu']
    if mt == 455:
        return ['nu-delayed']
    if mt == 456:
        return ['nu-prompt']
    if mt == CHI_MT:
        return ['chi-constrained']
    return [REACTION_NAME.get(mt, f'MT={mt}')]


def _summary_mt(mt):
    """Return the summary reaction that includes a partial reaction"""
    if 51 <= mt <= 91:
        return 4
    if mt in (19, 20, 21, 38):
        return 18
    for summary, start in ((103, 600), (104, 650), (105, 700), (106, 750),
                           (107, 800)):
        if start <= mt < start + 50:
            return summary
    return mt


def _primary_mts(cov):
    """Select a non-redundant set of covariance keys for a nuclide"""
    mts = set(cov.mts)
    mts -= {1, 3, 27, 101}

    # Partial reactions are represented by their summary reaction when it has
    # covariances
    mts -= {mt for mt in mts if _summary_mt(mt) != mt and _summary_mt(mt) in mts}

    # Lumped covariances are redundant when the summary reactions of all of
    # their members have covariances; otherwise their members are represented
    # by the lump
    for lump in [mt for mt in mts if mt in _LUMPED_MTS]:
        members = [mt for mt, l in cov.lumped.items() if l == lump]
        if members and all(_summary_mt(m) in mts and _summary_mt(m) != m
                           for m in members):
            mts.discard(lump)
        else:
            mts -= set(members)

    if 452 in mts:
        mts -= {455, 456}
    return sorted(mts)


@dataclass
class UncertaintyContribution:
    """Contribution of a pair of reactions to the variance of a response.

    Attributes
    ----------
    nuclide : str
        Name of the nuclide
    reaction1, reaction2 : str
        Labels of the reactions
    variance : float
        Contribution to the relative variance. For a pair of distinct
        reactions, this includes both orderings of the pair and may be
        negative.
    variance_std_dev : float
        Standard deviation of the variance contribution due to the
        statistical uncertainty of the sensitivity coefficients
    bias : float
        Expected increase of the variance contribution caused by the
        statistical noise of the sensitivity coefficients

    """
    nuclide: str
    reaction1: str
    reaction2: str
    variance: float
    variance_std_dev: float = 0.0
    bias: float = 0.0

    @property
    def std_dev(self):
        """Signed square root of the variance contribution"""
        return np.sign(self.variance) * np.sqrt(abs(self.variance))


@dataclass
class Uncertainty:
    """Relative uncertainty of a response due to nuclear data.

    .. versionadded:: 0.16.1

    Attributes
    ----------
    variance : float
        Relative variance of the response
    std_dev : float
        Relative standard deviation of the response
    variance_std_dev : float
        Standard deviation of the relative variance due to the statistical
        uncertainty of the sensitivity coefficients
    std_dev_uncertainty : float
        Standard deviation of the relative standard deviation due to the
        statistical uncertainty of the sensitivity coefficients
    bias : float
        Expected increase of the relative variance caused by the statistical
        noise of the sensitivity coefficients, :math:`\sum_x \sum_g C_{xx,gg}
        \sigma^2(S_{x,g})`. Statistical noise always increases the variance on
        average, which matters when sensitivities have large relative
        uncertainties.
    unbiased_variance : float
        Relative variance with the bias subtracted
    unbiased_std_dev : float
        Square root of the unbiased variance (0 if it is negative)
    contributions : list of UncertaintyContribution
        Contributions of pairs of reactions, sorted by decreasing magnitude
    missing_sensitivities : list of tuple
        (nuclide, reaction) pairs with covariances but no sensitivity
        coefficients
    missing_covariances : list of tuple
        (nuclide, reaction) pairs with sensitivity coefficients that were not
        used because no covariances are available for them

    """
    variance: float
    contributions: list = field(default_factory=list)
    missing_sensitivities: list = field(default_factory=list)
    missing_covariances: list = field(default_factory=list)
    variance_std_dev: float = 0.0
    bias: float = 0.0

    @property
    def std_dev(self):
        return np.sqrt(max(self.variance, 0.0))

    @property
    def std_dev_uncertainty(self):
        if self.variance <= 0.0:
            return 0.0
        return self.variance_std_dev / (2.0 * self.std_dev)

    @property
    def unbiased_variance(self):
        return self.variance - self.bias

    @property
    def unbiased_std_dev(self):
        return np.sqrt(max(self.unbiased_variance, 0.0))

    def by_nuclide(self) -> dict:
        """Return the variance contributions summed over each nuclide"""
        out = {}
        for c in self.contributions:
            out[c.nuclide] = out.get(c.nuclide, 0.0) + c.variance
        return out


def propagate_uncertainty(coefficients: Mapping, covariances: Mapping,
                          energies=None) -> Uncertainty:
    """Propagate nuclear data covariances to the uncertainty of a response.

    The relative variance of the response is computed with the sandwich rule,

    .. math::

        \\left ( \\frac{\\sigma_R}{R} \\right )^2 = \\sum_{x,y} S_x^T C_{xy}
        S_y,

    where :math:`S_x` is the vector of relative sensitivity coefficients to
    reaction :math:`x` in each energy group and :math:`C_{xy}` is the relative
    covariance matrix between reactions :math:`x` and :math:`y` of the same
    nuclide. A non-redundant set of reactions is used for each nuclide: the
    total, nonelastic, and absorption cross sections are excluded, inelastic
    levels are represented by the total inelastic cross section (MT 4) when
    its covariances are available, and the total fission neutron multiplicity
    (MT 452) is preferred to its prompt and delayed components.

    Sensitivity coefficients are matched to covariances by name: '(n,elastic)'
    or 'elastic' for MT 2, '(n,level)' for MT 4, 'fission' for MT 18, 'nu' for
    MT 452, 'nu-prompt' and 'nu-delayed' for MT 456 and 455, the sum of the
    constrained prompt and delayed fission spectrum sensitivities for the
    fission spectrum, and reaction names such as '(n,2n)' or '(n,gamma)'
    otherwise. Note that the 'inelastic' and 'capture' sensitivities combine
    several reactions and are not used.

    .. versionadded:: 0.16.1

    Parameters
    ----------
    coefficients : dict
        Dictionary mapping (nuclide, reaction) to sensitivity coefficients in
        each energy group, as returned by
        :meth:`openmc.StatePoint.get_keff_sensitivities` or
        :meth:`openmc.StatePoint.get_kinetics_sensitivities`. If the values
        have uncertainties, these are propagated to the variance to first
        order assuming that the coefficients of different groups and
        reactions are uncorrelated, and the bias of the variance due to
        statistical noise is estimated.
    covariances : dict
        Dictionary mapping nuclide names to :class:`MultigroupCovariance`
    energies : Iterable of float, optional
        Energy group boundaries of the sensitivity coefficients. If given, they
        are checked against the group structure of the covariances.

    Returns
    -------
    Uncertainty
        Relative uncertainty of the response and its contributions

    """
    def split(values):
        """Return nominal values and standard deviations"""
        values = np.asarray(values)
        if values.dtype == object:
            return (np.array([getattr(v, 'n', v) for v in values], dtype=float),
                    np.array([getattr(v, 's', 0.0) for v in values],
                             dtype=float))
        return values.astype(float), np.zeros(len(values))

    def add(parts):
        """Sum sensitivities assuming uncorrelated uncertainties"""
        return (sum(p[0] for p in parts),
                np.sqrt(sum(p[1]**2 for p in parts)))

    nuclides = {nuc for nuc, _ in coefficients}
    result = Uncertainty(0.0)
    used = set()

    for nuc in sorted(nuclides):
        if nuc not in covariances:
            result.missing_covariances.extend(
                (n, rx) for n, rx in coefficients if n == nuc)
            continue
        cov = covariances[nuc]
        if energies is not None and not np.allclose(
                cov.energies, energies, rtol=1e-4):
            raise ValueError(f'The group structure of the covariances of {nuc} '
                             'does not match that of the sensitivities.')

        def sensitivity(mt):
            if mt == CHI_MT:
                names = [n for n in ('chi-prompt-constrained',
                                     'chi-delayed-constrained')
                         if (nuc, n) in coefficients]
                if not names:
                    return None
                used.update((nuc, n) for n in names)
                used.update((nuc, n.replace('-constrained', ''))
                            for n in names)
                return add([split(coefficients[nuc, n]) for n in names])
            if mt in _LUMPED_MTS:
                # Sensitivities of charged-particle levels are only available
                # for the summary reaction, which is used for all its levels
                members = {m if 51 <= m <= 91 else _summary_mt(m)
                           for m, lump in cov.lumped.items() if lump == mt}
                parts = [sensitivity(m) for m in sorted(members)]
                parts = [p for p in parts if p is not None]
                return add(parts) if parts else None
            for name in _sensitivity_names(mt):
                if (nuc, name) in coefficients:
                    used.add((nuc, name))
                    if name == 'nu':
                        used.update({(nuc, 'nu-prompt'), (nuc, 'nu-delayed')})
                    return split(coefficients[nuc, name])
            return None

        mts = _primary_mts(cov)
        sens = {}
        for mt in mts:
            s = sensitivity(mt)
            if s is None:
                result.missing_sensitivities.append((nuc, _label(mt)))
            else:
                if len(s[0]) != cov.num_groups:
                    raise ValueError(
                        f'Sensitivities of {nuc} have {len(s[0])} groups but '
                        f'covariances have {cov.num_groups}.')
                sens[mt] = s

        # Derivative of the total variance with respect to each sensitivity,
        # used to propagate statistical uncertainties of the sensitivities
        keys = list(sens)
        gradient = {a: np.zeros(cov.num_groups) for a in keys}
        for i, a in enumerate(keys):
            for b in keys[i:]:
                matrix = cov.get(a, b)
                if matrix is None:
                    continue
                (s_a, sd_a), (s_b, sd_b) = sens[a], sens[b]
                var = s_a @ matrix @ s_b
                if a == b:
                    grad_a = 2.0 * matrix @ s_a
                    var_var = np.sum((grad_a * sd_a)**2)
                    bias = np.sum(np.diag(matrix) * sd_a**2)
                    gradient[a] += grad_a
                else:
                    var *= 2.0
                    grad_a = 2.0 * matrix @ s_b
                    grad_b = 2.0 * matrix.T @ s_a
                    var_var = (np.sum((grad_a * sd_a)**2) +
                               np.sum((grad_b * sd_b)**2))
                    bias = 0.0
                    gradient[a] += grad_a
                    gradient[b] += grad_b
                result.variance += var
                result.bias += bias
                result.contributions.append(UncertaintyContribution(
                    nuc, _label(a), _label(b), var, np.sqrt(var_var), bias))

        var_var = sum(np.sum((gradient[a] * sens[a][1])**2) for a in keys)
        result.variance_std_dev = np.sqrt(
            result.variance_std_dev**2 + var_var)

    # Sensitivities that were not used, excluding combined ones
    combined = {'inelastic', 'capture', 'scatter', 'total', 'nu',
                'chi-prompt', 'chi-delayed', 'chi-prompt-constrained',
                'chi-delayed-constrained', 'elastic', '(n,elastic)',
                'fission', '(n,fission)', 'nu-prompt', 'nu-delayed'}
    for key in coefficients:
        nuc, name = key
        if key in used or name in combined or nuc not in covariances:
            continue
        # Inelastic levels are included in the total inelastic cross section
        if 51 <= REACTION_MT.get(name, 0) <= 91 and (nuc, '(n,level)') in used:
            continue
        result.missing_covariances.append(key)

    result.contributions.sort(key=lambda c: -abs(c.variance))
    return result
