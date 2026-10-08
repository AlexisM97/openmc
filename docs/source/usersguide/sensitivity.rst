.. _sensitivity:

=========================
Sensitivity coefficients
=========================

OpenMC can estimate the sensitivity coefficients of the effective
multiplication factor and of the adjoint-weighted kinetics parameters
:math:`\beta_{\text{eff}}` and :math:`\Lambda_{\text{eff}}` to nuclear data,

.. math::
    :label: sensitivity_coefficient

    S_{k,x} = \frac{\delta k / k}{\delta x / x},

where :math:`x` is a cross section, a fission neutron multiplicity, or a
fission spectrum of a given nuclide in a given energy group, and similarly for
the kinetics parameters. The sensitivity
coefficients are calculated using the collision history-based approach
[Aufiero_2015]_ implemented in Serpent 2, in which the importance of
each neutron is estimated with the iterated fission probability (IFP) method
(see :ref:`kinetics`). The implementation is limited to continuous-energy
eigenvalue calculations.

------
Method
------

A relative perturbation :math:`\epsilon` of the nuclear data :math:`x` changes
the probability of every neutron history. The derivative of the logarithm of
the probability of a history with respect to :math:`\epsilon` is a sum of
contributions from the events of the history:

- Every collision with nuclide :math:`n` in which reaction :math:`x` is sampled
  contributes :math:`+1`.
- Every flight path of length :math:`s` contributes :math:`-\Sigma_x s`, the
  derivative of the probability of travelling the path without colliding. An
  unbiased estimate of this quantity is obtained by sampling pseudo-collisions
  along the path with a rate equal to the total macroscopic cross section and
  scoring :math:`-1` for the nuclide and reaction sampled at each
  pseudo-collision.
- Every fission neutron contributes :math:`+1` for the fission cross section of
  the fissioned nuclide, :math:`+1` for its prompt or delayed multiplicity, and
  :math:`+1` for its prompt or delayed fission spectrum in the outgoing energy
  group.

These events are recorded in the history of each neutron and passed on to its
fission progeny along with the events of its ancestors, in the same way that
lifetimes and delayed groups are passed on with the IFP method. Following
first-order perturbation theory, the sensitivity of :math:`k` is the
adjoint-weighted sum of the events of a single generation. The adjoint
weighting is obtained by attributing the events of the ancestor that lived
:math:`N_{\text{gen}}` generations ago to the fission events of each neutron,
where :math:`N_{\text{gen}}` is set with the ``ifp_n_generation`` setting.

Because the elastic, inelastic, capture, and fission cross sections form a
partition of the total cross section, the sensitivities to the total and
scattering cross sections are obtained by summing their components. Inelastic
scattering includes all non-elastic scattering reactions including
:math:`(n,xn)` reactions, and elastic scattering includes thermal scattering
from :math:`S(\alpha,\beta)` tables. The sensitivity to the fission spectrum is
first computed without constraint and then constrained to preserve the
normalization of the spectrum using

.. math::
    :label: constrained_chi

    S^{\text{constrained}}_{\chi,g} = S_{\chi,g} - \chi_g \sum_{g'} S_{\chi,g'},

where :math:`\chi_g` is the fraction of fission neutrons of the nuclide born in
group :math:`g` in the problem.

-------------------------------------
Obtaining sensitivity coefficients
-------------------------------------

The :meth:`openmc.Model.add_keff_sensitivity_tallies` method adds all tallies
needed to compute the sensitivity coefficients for a list of nuclides and an
energy group structure::

    model.settings.ifp_n_generation = 10
    model.add_keff_sensitivity_tallies(
        nuclides=['U235', 'U238'],
        energies='ECCO-33',
        reactions=['elastic', 'inelastic', 'capture', 'fission', 'nu-prompt',
                   'nu-delayed', 'chi-prompt', 'chi-delayed']
    )

Alternatively, the tallies can be defined manually using a
:class:`openmc.SensitivityFilter` with the ``keff-sensitivity-numerator``
score, along with a tally of the ``ifp-denominator`` score::

    sens_filter = openmc.SensitivityFilter(
        ['U235'], ['fission', 'nu-prompt'], [1e-5, 0.625, 1e5, 2e7])
    sens_tally = openmc.Tally()
    sens_tally.filters = [sens_filter]
    sens_tally.scores = ['keff-sensitivity-numerator']

    denom_tally = openmc.Tally()
    denom_tally.scores = ['ifp-denominator']

The sensitivity filter determines the bin of each event recorded in the
ancestry of a neutron, so it cannot be combined with other filters or with
nuclide bins. The sensitivity coefficients are obtained by dividing the
``keff-sensitivity-numerator`` score by the ``ifp-denominator`` score. They can
be retrieved directly from a statepoint file using the
:meth:`openmc.StatePoint.get_keff_sensitivities` method, which also computes
the combined sensitivities ('scatter', 'total', and 'nu') and the constrained
fission spectrum sensitivities::

    with openmc.StatePoint(output_path) as sp:
        energies, coefficients = sp.get_keff_sensitivities()
    s_fission = coefficients['U235', 'fission']  # one value per group
    s_total = coefficients['U235', 'total'].sum()  # energy-integrated

.. note:: Each fission neutron stores the events of its last
    ``ifp_n_generation`` ancestors. The memory footprint therefore increases
    with ``ifp_n_generation``, the number of particles per generation, and the
    number of nuclides and reactions that are perturbed, since only events
    for the perturbed nuclides and reactions are recorded.

-------------------------------------------
Sensitivity of kinetics parameters
-------------------------------------------

The effective delayed neutron fraction and the effective generation time are
ratios of adjoint-weighted quantities,

.. math::
    :label: kinetics_ratios

    \beta_{\text{eff}} = \frac{E[X]}{E[Y]}, \qquad
    \Lambda_{\text{eff}} = \frac{E[T]}{k \, E[Y]},

where, for each source neutron of an active generation, :math:`Y = 1`,
:math:`X` is 1 if its ancestor :math:`N_{\text{gen}}` generations old was
born from a delayed neutron precursor, and :math:`T` is the lifetime of that
ancestor. A perturbation of nuclear data changes both the importance of the
ancestor, which is carried by the generations that follow it, and the
fundamental-mode source from which the ancestor is sampled, which converges
over the generations that precede it. The sensitivity of each expectation is
obtained by weighting it with the sum :math:`D_x` of the events of all
ancestors of the neutron over :math:`N_{\text{gen}} + 1 + N_{\text{latent}}`
generations:

.. math::
    :label: kinetics_sensitivity

    S_{\beta,x} = \frac{E[X D_x]}{E[X]} - \frac{E[Y D_x]}{E[Y]}, \qquad
    S_{\Lambda,x} = \frac{E[T D_x]}{E[T]} - \frac{E[Y D_x]}{E[Y]}
    - S_{k,x}.

The normalization of each generation by :math:`k` affects all three
expectations equally and cancels in these differences. To reduce the variance
of the differences, the numerators are tallied as :math:`E[(X - \hat{\beta})
D_x]` and :math:`E[(T - \hat{T}) D_x]`, where :math:`\hat{\beta}` and
:math:`\hat{T}` are estimated during inactive generations, and the centering is
removed exactly when the results are processed.

:math:`N_{\text{gen}}` is set with ``ifp_n_generation`` and
:math:`N_{\text{latent}}` with ``sensitivity_latent_generations`` (10 by
default). The number of latent generations should be large enough for the
fundamental-mode source to converge after a perturbation, which requires more
generations for loosely coupled systems with a dominance ratio close to one.
The total number of generations must not exceed the number of inactive
batches::

    model.settings.inactive = 40
    model.settings.ifp_n_generation = 10
    model.settings.sensitivity_latent_generations = 10
    model.add_kinetics_sensitivity_tallies(['U235', 'U238'], 'ECCO-33')

    ...

    with openmc.StatePoint(output_path) as sp:
        result = sp.get_kinetics_sensitivities()
    print(result.beta_effective, result.generation_time)
    s_beta = result.beta_coefficients['U238', 'nu-delayed']
    s_gen_time = result.generation_time_coefficients['U235', 'total']

:meth:`openmc.Model.add_kinetics_sensitivity_tallies` also adds the tallies for
:math:`k_{\text{eff}}` sensitivities, which are needed for the generation
time. The ``kinetics-population``, ``kinetics-delayed`` and
``kinetics-lifetime`` scores estimate :math:`E[Y]`, :math:`E[X]` and
:math:`E[T]`, while the ``population-sensitivity-numerator``,
``beta-sensitivity-numerator`` and ``lifetime-sensitivity-numerator`` scores,
which require a :class:`openmc.SensitivityFilter`, estimate the numerators of
Eq. :eq:`kinetics_sensitivity`.

Because delayed neutrons are rare, the statistical uncertainty of
:math:`\beta_{\text{eff}}` sensitivities is much larger than that of
:math:`k_{\text{eff}}` sensitivities for the same number of histories.

-----------
Limitations
-----------

- Sensitivity coefficients can be computed for :math:`k_{\text{eff}}`, the total
  :math:`\beta_{\text{eff}}`, and :math:`\Lambda_{\text{eff}}`.
  Sensitivities of group-wise :math:`\beta_{\text{eff}}` are not supported.
- Perturbations of scattering angular and secondary energy distributions are
  not supported.
- The method cannot be used in multigroup mode, in fixed source calculations,
  with a shared secondary bank, or with NCrystal materials.

.. only:: html

   .. rubric:: References

.. [Aufiero_2015] M. Aufiero, A. Bidaud, M. Hursin, J. Leppänen, G. Palmiotti,
    S. Pelloni, and P. Rubiolo, "A collision history-based approach to
    sensitivity/perturbation calculations in the continuous energy Monte Carlo
    code SERPENT", Annals of Nuclear Energy, 85, 245-258 (2015).
