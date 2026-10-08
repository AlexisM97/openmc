#include "openmc/sensitivity.h"

#include <algorithm> // for min, max
#include <climits>   // for INT_MAX
#include <cmath>     // for log

#include "openmc/bank.h"
#include "openmc/constants.h"
#include "openmc/container_util.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/filter_sensitivity.h"
#include "openmc/tallies/tally.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

const array<const char*, N_SENSITIVITY_REACTIONS> SENSITIVITY_REACTION_NAMES {
  "elastic", "inelastic", "capture", "fission", "nu-prompt", "nu-delayed",
  "chi-prompt", "chi-delayed"};

namespace model {

vector<char> sensitivity_nuclides;
array<bool, N_SENSITIVITY_REACTIONS> sensitivity_reactions {};

} // namespace model

namespace simulation {

vector<SensitivityChain> sensitivity_source_bank;
vector<SensitivityChain> sensitivity_fission_bank;
vector<SensitivityChain> sensitivity_temp_bank;
array<double, 2> kinetics_reference {};

} // namespace simulation

//==============================================================================
// Helper functions
//==============================================================================

namespace {

//! Weighted sums of the population, delayed indicator, and lifetime of the
//! ancestors of source neutrons during inactive generations
array<double, 3> kinetics_reference_sums {};

//! Determine the positions of the end markers of the links in a chain
//!
//! \param[in] chain Chain of ancestors
//! \param[out] ends Position of the end marker of each link
void find_link_ends(const SensitivityChain& chain, vector<int64_t>& ends)
{
  ends.clear();
  for (int64_t i = 0; i < chain.size(); ++i) {
    if (chain[i].is_link_end())
      ends.push_back(i);
  }
}

//! Accumulate the events of a range of a chain into a dense array of filter
//! bins, keeping track of the bins that were touched
void accumulate_events(const SensitivityFilter& filt,
  SensitivityChain::const_iterator begin, SensitivityChain::const_iterator end,
  vector<int>& values, vector<char>& seen, vector<int>& touched)
{
  for (auto it = begin; it != end; ++it) {
    if (it->is_link_end())
      continue;
    int bin = filt.get_bin(*it);
    if (bin == C_NONE)
      continue;
    if (!seen[bin]) {
      seen[bin] = true;
      touched.push_back(bin);
    }
    values[bin] += it->sign();
  }
}

//! Add an event to a link if the nuclide and reaction are being tracked
void add_event(SensitivityLink& link, int i_nuclide, SensitivityReaction rx,
  double E, int sign)
{
  int i_rx = static_cast<int>(rx);
  if (!model::sensitivity_nuclides[i_nuclide] ||
      !model::sensitivity_reactions[i_rx])
    return;
  int32_t code = 1 + i_nuclide * N_SENSITIVITY_REACTIONS + i_rx;
  link.push_back({sign * code, static_cast<float>(E)});
}

//! Sample the reaction channel of a pseudo-collision with a given nuclide
//! using the same partition of the total cross section as the one used when
//! sampling real collisions (see sample_neutron_reaction and scatter).
SensitivityReaction sample_reaction_channel(
  Particle& p, int i_nuclide, uint64_t* seed)
{
  const auto& micro = p.neutron_xs(i_nuclide);
  if (micro.elastic == CACHE_INVALID) {
    data::nuclides[i_nuclide]->calculate_elastic_xs(p);
  }

  double cutoff = prn(seed) * micro.total;
  double scatter = micro.total - micro.absorption;
  if (cutoff < micro.elastic) {
    return SensitivityReaction::ELASTIC;
  } else if (cutoff < scatter) {
    return SensitivityReaction::INELASTIC;
  } else if (cutoff < scatter + micro.fission) {
    return SensitivityReaction::FISSION;
  } else {
    return SensitivityReaction::CAPTURE;
  }
}

} // namespace

//==============================================================================
// Non-member functions
//==============================================================================

int n_sensitivity_links(const SensitivityChain& chain)
{
  return std::count_if(chain.begin(), chain.end(),
    [](const SensitivityEvent& e) { return e.is_link_end(); });
}

int sensitivity_chain_length()
{
  int n = settings::ifp_n_generation;
  if (settings::kinetics_sensitivity_on) {
    int n_latent = settings::sensitivity_latent_generations >= 0
                     ? settings::sensitivity_latent_generations
                     : DEFAULT_SENSITIVITY_LATENT_GENERATIONS;
    n += 1 + n_latent;
  }
  return n;
}

bool is_kinetics_score(int score)
{
  switch (score) {
  case SCORE_KINETICS_POPULATION:
  case SCORE_KINETICS_DELAYED:
  case SCORE_KINETICS_LIFETIME:
  case SCORE_POPULATION_SENSITIVITY_NUM:
  case SCORE_BETA_SENSITIVITY_NUM:
  case SCORE_LIFETIME_SENSITIVITY_NUM:
    return true;
  default:
    return false;
  }
}

void check_sensitivity_settings()
{
  if (settings::run_mode != RunMode::EIGENVALUE) {
    fatal_error(
      "Sensitivity coefficients can only be calculated in an eigenvalue "
      "calculation.");
  }
  if (!settings::run_CE) {
    fatal_error("Sensitivity coefficients can only be calculated in "
                "continuous-energy mode.");
  }
  if (settings::use_shared_secondary_bank) {
    fatal_error("Sensitivity coefficients cannot be calculated with a shared "
                "secondary bank.");
  }
  for (const auto& mat : model::materials) {
    if (mat->ncrystal_mat()) {
      fatal_error(fmt::format("Sensitivity coefficients cannot be calculated "
                              "with NCrystal materials (material {}).",
        mat->id()));
    }
  }
}

void initialize_sensitivity()
{
  model::sensitivity_nuclides.assign(data::nuclides.size(), false);
  model::sensitivity_reactions.fill(false);
  simulation::kinetics_reference.fill(0.0);
  kinetics_reference_sums.fill(0.0);

  for (const auto& tally : model::tallies) {
    bool has_keff = contains(tally->scores_, SCORE_KEFF_SENSITIVITY_NUM);
    bool has_binned = false;
    bool has_scalar = false;
    for (auto score : tally->scores_) {
      has_binned |= (score == SCORE_POPULATION_SENSITIVITY_NUM ||
                     score == SCORE_BETA_SENSITIVITY_NUM ||
                     score == SCORE_LIFETIME_SENSITIVITY_NUM);
      has_scalar |=
        (score == SCORE_KINETICS_POPULATION ||
          score == SCORE_KINETICS_DELAYED || score == SCORE_KINETICS_LIFETIME);
    }
    if (!has_keff && !has_binned && !has_scalar)
      continue;

    if (tally->nuclides_.size() != 1 || tally->nuclides_[0] != -1) {
      fatal_error(fmt::format(
        "Tally {} with sensitivity or kinetics scores cannot have nuclide "
        "bins. Perturbed nuclides are specified on the sensitivity filter.",
        tally->id_));
    }

    // Scores of different kinds are scored at different points of a history
    // and cannot be mixed
    int n_kinds = has_keff + has_binned + has_scalar;
    int n_kind_scores = 0;
    for (auto score : tally->scores_) {
      if (score == SCORE_KEFF_SENSITIVITY_NUM || is_kinetics_score(score))
        ++n_kind_scores;
    }
    if (n_kinds > 1 || n_kind_scores != tally->scores_.size() ||
        (has_keff && tally->scores_.size() != 1)) {
      fatal_error(fmt::format(
        "Tally {} mixes sensitivity or kinetics scores with other scores. "
        "'keff-sensitivity-numerator' must be the only score of its tally, "
        "and binned and unbinned kinetics scores require separate tallies.",
        tally->id_));
    }

    if (has_scalar) {
      if (!tally->filters().empty()) {
        fatal_error(fmt::format(
          "Tally {} with unbinned kinetics scores cannot have filters.",
          tally->id_));
      }
      continue;
    }

    // The sensitivity filter determines which bin each event is scored to, so
    // it cannot be combined with other filters
    const SensitivityFilter* filt = nullptr;
    if (tally->filters().size() == 1) {
      filt = dynamic_cast<const SensitivityFilter*>(
        model::tally_filters[tally->filters(0)].get());
    }
    if (!filt) {
      fatal_error(fmt::format("Tally {} with sensitivity scores must have a "
                              "single sensitivity filter.",
        tally->id_));
    }

    for (auto i_nuclide : filt->nuclides()) {
      model::sensitivity_nuclides[i_nuclide] = true;
    }
    for (auto rx : filt->reactions()) {
      model::sensitivity_reactions[rx] = true;
    }
  }

  if (sensitivity_chain_length() > settings::n_inactive) {
    fatal_error(fmt::format(
      "Sensitivity coefficients require {} generations of ancestors, which "
      "must be lower than or equal to the number of inactive batches. The "
      "number of generations is ifp_n_generation, plus 1 + "
      "sensitivity_latent_generations for kinetics parameters.",
      sensitivity_chain_length()));
  }
}

void resize_sensitivity_banks()
{
  simulation::sensitivity_source_bank.clear();
  simulation::sensitivity_source_bank.resize(simulation::work_per_rank);
  simulation::sensitivity_fission_bank.clear();
  simulation::sensitivity_fission_bank.resize(3 * simulation::work_per_rank);
  simulation::sensitivity_temp_bank.clear();
  simulation::sensitivity_temp_bank.resize(3 * simulation::work_per_rank);
}

void free_memory_sensitivity()
{
  simulation::sensitivity_source_bank.clear();
  simulation::sensitivity_fission_bank.clear();
  simulation::sensitivity_temp_bank.clear();
  model::sensitivity_nuclides.clear();
  model::sensitivity_reactions.fill(false);
}

void record_sensitivity_event(
  Particle& p, int i_nuclide, SensitivityReaction rx, double E, int sign)
{
  add_event(p.sensitivity_events(), i_nuclide, rx, E, sign);
}

void sample_sensitivity_flight(Particle& p, double distance)
{
  double sigma_t = p.macro_xs().total;
  if (p.material() == MATERIAL_VOID || sigma_t <= 0.0)
    return;

  const auto& mat = *model::materials[p.material()];
  uint64_t* seed = &p.seeds(STREAM_SENSITIVITY);

  // Sample pseudo-collisions along the path with rate equal to the total
  // macroscopic cross section
  double s = -std::log(prn(seed)) / sigma_t;
  while (s < distance) {
    // Sample a nuclide in proportion to its macroscopic total cross section
    double cutoff = prn(seed) * sigma_t;
    double prob = 0.0;
    int i_nuclide = mat.nuclide_.back();
    for (int i = 0; i < mat.nuclide_.size(); ++i) {
      int j_nuclide = mat.nuclide_[i];
      prob +=
        mat.atom_density(i, p.density_mult()) * p.neutron_xs(j_nuclide).total;
      if (prob >= cutoff) {
        i_nuclide = j_nuclide;
        break;
      }
    }

    // Sample the reaction channel and score a negative event
    if (model::sensitivity_nuclides[i_nuclide]) {
      auto rx = sample_reaction_channel(p, i_nuclide, seed);
      record_sensitivity_event(p, i_nuclide, rx, p.E(), -1);
    }

    s -= std::log(prn(seed)) / sigma_t;
  }
}

void bank_sensitivity_fission_site(
  const Particle& p, int i_nuclide, const SourceSite& site, int64_t idx)
{
  // Copy the links of the ancestors, dropping the oldest one if the chain
  // would otherwise exceed its maximum length
  const auto& parent_chain =
    simulation::sensitivity_source_bank[p.current_work()];
  auto start = parent_chain.begin();
  if (n_sensitivity_links(parent_chain) >= sensitivity_chain_length()) {
    start = std::find_if(start, parent_chain.end(),
              [](const SensitivityEvent& e) { return e.is_link_end(); }) +
            1;
  }
  auto& chain = simulation::sensitivity_fission_bank[idx];
  chain.assign(start, parent_chain.end());

  // The link from the parent to this fission site consists of all events of
  // the parent so far plus the events associated with the production of the
  // fission neutron
  const auto& events = p.sensitivity_events();
  chain.insert(chain.end(), events.begin(), events.end());
  double E_in = p.E();
  bool delayed = site.delayed_group > 0;
  add_event(chain, i_nuclide, SensitivityReaction::FISSION, E_in, 1);
  add_event(chain, i_nuclide,
    delayed ? SensitivityReaction::NU_DELAYED : SensitivityReaction::NU_PROMPT,
    E_in, 1);
  add_event(chain, i_nuclide,
    delayed ? SensitivityReaction::CHI_DELAYED
            : SensitivityReaction::CHI_PROMPT,
    site.E, 1);
  chain.push_back(sensitivity_link_end(site.delayed_group, p.lifetime()));
}

void save_sensitivity_secondary(Particle& p)
{
  int64_t bank_index = p.local_secondary_bank().size() - 1;
  p.sensitivity_secondaries().push_back({bank_index, p.sensitivity_events()});
}

void restore_sensitivity_secondary(Particle& p)
{
  auto& secondaries = p.sensitivity_secondaries();
  int64_t bank_index = p.local_secondary_bank().size() - 1;
  if (!secondaries.empty() && secondaries.back().bank_index == bank_index) {
    p.sensitivity_events() = std::move(secondaries.back().events);
    secondaries.pop_back();
  }
}

void score_keff_sensitivity(const Particle& p, int i_tally, int score_index)
{
  const auto& chain = simulation::sensitivity_source_bank[p.current_work()];
  int n_links = sensitivity_chain_length();
  if (n_sensitivity_links(chain) != n_links)
    return;

  // The importance of an ancestor is proportional to the weight of its
  // descendants ifp_n_generation - 1 generations later, so the events of the
  // link that produced that ancestor are weighted by the current neutron's
  // contribution. Older links are only stored for kinetics parameters.
  int i_link = n_links - settings::ifp_n_generation;
  auto it = chain.begin();
  for (int i = 0; i < i_link; ++i) {
    it = std::find_if(it, chain.end(), [](const SensitivityEvent& e) {
      return e.is_link_end();
    }) + 1;
  }

  auto& tally = *model::tallies[i_tally];
  const auto& filt = *static_cast<const SensitivityFilter*>(
    model::tally_filters[tally.filters(0)].get());
  double wgt = p.wgt_last();
  for (; it != chain.end() && !it->is_link_end(); ++it) {
    int bin = filt.get_bin(*it);
    if (bin == C_NONE)
      continue;
#pragma omp atomic
    tally.results_(bin, score_index, TallyResult::VALUE) += it->sign() * wgt;
  }
}

void score_kinetics_sensitivity(const Particle& p)
{
  const auto& chain = simulation::sensitivity_source_bank[p.current_work()];
  int n_links = sensitivity_chain_length();

  thread_local vector<int64_t> ends;
  find_link_ends(chain, ends);
  if (ends.size() != n_links)
    return;

  // Link i produced the ancestor that is n_links - 1 - i generations old. The
  // delayed group of the ancestor ifp_n_generation generations old is stored
  // at the end of the link that produced it, and its lifetime is stored at
  // the end of the next link, consistent with the IFP scores.
  int n_gen = settings::ifp_n_generation;
  double delayed =
    chain[ends[n_links - 1 - n_gen]].delayed_group() > 0 ? 1.0 : 0.0;
  double lifetime = chain[ends[n_links - n_gen]].lifetime();
  double wgt = p.wgt();

  // During inactive generations, accumulate reference values used to center
  // the scores of the active generations
  if (simulation::current_batch <= settings::n_inactive) {
#pragma omp critical(kinetics_reference)
    {
      kinetics_reference_sums[0] += wgt;
      kinetics_reference_sums[1] += wgt * delayed;
      kinetics_reference_sums[2] += wgt * lifetime;
    }
    return;
  }

  double beta_ref = simulation::kinetics_reference[0];
  double lifetime_ref = simulation::kinetics_reference[1];

  thread_local vector<int> values;
  thread_local vector<char> seen;
  thread_local vector<int> touched;
  for (auto i_tally : model::active_birth_tallies) {
    auto& tally = *model::tallies[i_tally];

    // Unbinned scores
    if (tally.filters().empty()) {
      for (int i = 0; i < tally.scores_.size(); ++i) {
        double score = 0.0;
        switch (tally.scores_[i]) {
        case SCORE_KINETICS_POPULATION:
          score = wgt;
          break;
        case SCORE_KINETICS_DELAYED:
          score = wgt * delayed;
          break;
        case SCORE_KINETICS_LIFETIME:
          score = wgt * lifetime;
          break;
        }
#pragma omp atomic
        tally.results_(0, i, TallyResult::VALUE) += score;
      }
      continue;
    }

    // Sum the events of all links of the chain in each filter bin
    const auto& filt = *static_cast<const SensitivityFilter*>(
      model::tally_filters[tally.filters(0)].get());
    values.assign(filt.n_bins(), 0);
    seen.assign(filt.n_bins(), false);
    touched.clear();
    accumulate_events(filt, chain.begin(), chain.end(), values, seen, touched);

    for (int i = 0; i < tally.scores_.size(); ++i) {
      double coeff = 0.0;
      switch (tally.scores_[i]) {
      case SCORE_POPULATION_SENSITIVITY_NUM:
        coeff = wgt;
        break;
      case SCORE_BETA_SENSITIVITY_NUM:
        coeff = wgt * (delayed - beta_ref);
        break;
      case SCORE_LIFETIME_SENSITIVITY_NUM:
        coeff = wgt * (lifetime - lifetime_ref);
        break;
      }
      for (auto bin : touched) {
        if (values[bin] == 0)
          continue;
#pragma omp atomic
        tally.results_(bin, i, TallyResult::VALUE) += coeff * values[bin];
      }
    }
  }
}

void finalize_kinetics_reference()
{
  auto sums = kinetics_reference_sums;
#ifdef OPENMC_MPI
  MPI_Allreduce(MPI_IN_PLACE, sums.data(), sums.size(), MPI_DOUBLE, MPI_SUM,
    mpi::intracomm);
#endif
  if (sums[0] > 0.0) {
    simulation::kinetics_reference[0] = sums[1] / sums[0];
    simulation::kinetics_reference[1] = sums[2] / sums[0];
  }
}

//==============================================================================
// Fission bank handling
//==============================================================================

void copy_sensitivity_from_fission_bank(int64_t i_bank, int64_t i_temp)
{
  simulation::sensitivity_temp_bank[i_temp] =
    simulation::sensitivity_fission_bank[i_bank];
}

void move_sensitivity_to_source_bank(int64_t i_temp, int64_t n, int64_t i_bank)
{
  for (int64_t i = 0; i < n; ++i) {
    simulation::sensitivity_source_bank[i_bank + i].swap(
      simulation::sensitivity_temp_bank[i_temp + i]);
  }
}

#ifdef OPENMC_MPI

void exchange_sensitivity_chains(const vector<SensitivityTransfer>& sends,
  const vector<SensitivityTransfer>& recvs)
{
  const auto& temp = simulation::sensitivity_temp_bank;
  auto& source_bank = simulation::sensitivity_source_bank;

  // First, communicate the number of events in each chain
  vector<MPI_Request> requests;
  vector<vector<int>> send_lengths(sends.size());
  for (int k = 0; k < sends.size(); ++k) {
    const auto& s = sends[k];
    auto& lengths = send_lengths[k];
    lengths.resize(s.n);
    for (int64_t i = 0; i < s.n; ++i) {
      lengths[i] = temp[s.index + i].size();
    }
    requests.emplace_back();
    MPI_Isend(lengths.data(), s.n, MPI_INT, s.neighbor, mpi::rank,
      mpi::intracomm, &requests.back());
  }

  vector<vector<int>> recv_lengths(recvs.size());
  for (int k = 0; k < recvs.size(); ++k) {
    const auto& r = recvs[k];
    recv_lengths[k].resize(r.n);
    requests.emplace_back();
    MPI_Irecv(recv_lengths[k].data(), r.n, MPI_INT, r.neighbor, r.neighbor,
      mpi::intracomm, &requests.back());
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  requests.clear();

  // Then, communicate the events themselves
  vector<vector<SensitivityEvent>> send_events(sends.size());
  for (int k = 0; k < sends.size(); ++k) {
    const auto& s = sends[k];
    auto& buffer = send_events[k];
    for (int64_t i = 0; i < s.n; ++i) {
      const auto& chain = temp[s.index + i];
      buffer.insert(buffer.end(), chain.begin(), chain.end());
    }
    if (buffer.size() * sizeof(SensitivityEvent) > INT_MAX) {
      fatal_error("Sensitivity data sent between processes is too large.");
    }
    requests.emplace_back();
    MPI_Isend(buffer.data(), buffer.size() * sizeof(SensitivityEvent), MPI_BYTE,
      s.neighbor, mpi::rank, mpi::intracomm, &requests.back());
  }

  vector<vector<SensitivityEvent>> recv_events(recvs.size());
  for (int k = 0; k < recvs.size(); ++k) {
    const auto& r = recvs[k];
    int64_t n_events = 0;
    for (auto length : recv_lengths[k]) {
      n_events += length;
    }
    if (n_events * sizeof(SensitivityEvent) > INT_MAX) {
      fatal_error("Sensitivity data received from processes is too large.");
    }
    recv_events[k].resize(n_events);
    requests.emplace_back();
    MPI_Irecv(recv_events[k].data(), n_events * sizeof(SensitivityEvent),
      MPI_BYTE, r.neighbor, r.neighbor, mpi::intracomm, &requests.back());
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  // Reconstruct the chains of the received sites
  for (int k = 0; k < recvs.size(); ++k) {
    const auto& r = recvs[k];
    auto event = recv_events[k].begin();
    for (int64_t i = 0; i < r.n; ++i) {
      int length = recv_lengths[k][i];
      source_bank[r.index + i].assign(event, event + length);
      event += length;
    }
  }
}

#endif

} // namespace openmc
