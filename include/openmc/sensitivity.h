//! \file sensitivity.h
//! \brief Collision-history based nuclear data sensitivity coefficients

#ifndef OPENMC_SENSITIVITY_H
#define OPENMC_SENSITIVITY_H

#include <cstdint>
#include <string>

#include "openmc/array.h"
#include "openmc/vector.h"

namespace openmc {

class Particle;
struct SourceSite;

//==============================================================================
// Constants
//==============================================================================

//! Nuclear data that can be perturbed in a sensitivity calculation are
//! identified by channels. Channels below 1000 are the MT numbers of reaction
//! cross sections, normalized so that partial fission reactions map to MT 18
//! and charged-particle level reactions (MT 600-849) map to MT 103-107. Thermal
//! scattering is attributed to elastic scattering (MT 2). Channels 1000 and
//! above are fission neutron multiplicities and spectra.
constexpr int SENSITIVITY_NU_PROMPT {1000};   //!< Prompt fission multiplicity
constexpr int SENSITIVITY_NU_DELAYED {1001};  //!< Delayed fission multiplicity
constexpr int SENSITIVITY_CHI_PROMPT {1002};  //!< Prompt fission spectrum
constexpr int SENSITIVITY_CHI_DELAYED {1003}; //!< Delayed fission spectrum
constexpr int N_SENSITIVITY_CHANNELS {1004};

//! Normalize the MT number of a reaction to its sensitivity channel
//!
//! \param[in] mt MT number of a reaction
//! \return Sensitivity channel
int sensitivity_channel(int mt);

//! Determine the sensitivity channels corresponding to the name of perturbed
//! nuclear data. The names 'elastic', 'inelastic' (all non-elastic scattering
//! including (n,xn) reactions), 'capture' (all non-fission absorption) and
//! 'fission' form a partition of the total cross section. Reaction names
//! accepted by tallies (e.g., '(n,2n)' or '(n,gamma)') select the channels of
//! a single reaction, where '(n,level)' (MT 4) selects all inelastic levels.
//! The names 'nu-prompt', 'nu-delayed', 'chi-prompt' and 'chi-delayed' select
//! fission neutron multiplicities and spectra.
//!
//! \param[in] name Name of perturbed nuclear data
//! \return Sensitivity channels
vector<int> sensitivity_channels(const std::string& name);

//==============================================================================
//! An event that contributes to the derivative of a neutron history's
//! probability with respect to a relative perturbation of nuclear data.
//!
//! The nuclide, perturbed channel and sign of the event are packed into a
//! single integer. All events have a magnitude of one, so the sign indicates
//! whether the event increases or decreases the history probability.
//!
//! The end of a link in a chain of ancestors is marked by a special event whose
//! code stores the delayed group of the fission neutron produced at the end of
//! the link and whose energy field stores the lifetime of the parent neutron
//! when it produced the fission neutron.
//==============================================================================

constexpr int32_t SENSITIVITY_LINK_END_CODE {INT32_MIN};
constexpr int32_t SENSITIVITY_LINK_END_RANGE {64}; //!< codes reserved

struct SensitivityEvent {
  int32_t code; //!< sign * (1 + i_nuclide * N_SENSITIVITY_CHANNELS + channel)
  float E;      //!< incident energy, or outgoing energy for spectrum events

  bool is_link_end() const
  {
    return code < SENSITIVITY_LINK_END_CODE + SENSITIVITY_LINK_END_RANGE;
  }
  int sign() const { return code > 0 ? 1 : -1; }
  int index() const { return (code > 0 ? code : -code) - 1; }
  int nuclide() const { return index() / N_SENSITIVITY_CHANNELS; }
  int channel() const { return index() % N_SENSITIVITY_CHANNELS; }

  //! Delayed group of the fission neutron produced at the end of a link
  int delayed_group() const { return code - SENSITIVITY_LINK_END_CODE; }
  //! Lifetime of the parent neutron at the end of a link in [s]
  double lifetime() const { return E; }
};

//! Events accumulated by a single neutron from its birth until it creates a
//! given fission site
using SensitivityLink = vector<SensitivityEvent>;

//! Links of the direct ancestors of a neutron, ordered from oldest to newest.
//! The links are stored contiguously, each one followed by an end marker.
using SensitivityChain = vector<SensitivityEvent>;

//! Create the marker for the end of a link
//!
//! \param[in] delayed_group Delayed group of the fission neutron produced
//! \param[in] lifetime Lifetime of the parent neutron in [s]
//! \return End marker
inline SensitivityEvent sensitivity_link_end(int delayed_group, double lifetime)
{
  return {
    SENSITIVITY_LINK_END_CODE + delayed_group, static_cast<float>(lifetime)};
}

//! Number of links in a chain
//!
//! \param[in] chain Chain of ancestors
//! \return Number of links
int n_sensitivity_links(const SensitivityChain& chain);

//! Whether a tally score is scored when source neutrons are born for the
//! calculation of kinetics parameters and their sensitivities
//!
//! \param[in] score Tally score
//! \return Whether the score is a kinetics score
bool is_kinetics_score(int score);

//! Number of links stored in the chain of each neutron, which is
//! ifp_n_generation for k-eff sensitivities. Sensitivities of kinetics
//! parameters require the link that produced the ancestor ifp_n_generation
//! generations old as well as sensitivity_latent_generations earlier links.
int sensitivity_chain_length();

//! Event log saved for a secondary neutron so that it continues the history
//! of the particle that created it
struct SensitivitySecondary {
  int64_t bank_index;     //!< index in the particle's local secondary bank
  SensitivityLink events; //!< events of the parent at creation of secondary
};

//==============================================================================
// Global variables
//==============================================================================

namespace model {

//! Whether events for a given nuclide (indexed as data::nuclides) are recorded
extern vector<char> sensitivity_nuclides;

//! Whether events for a given sensitivity channel are recorded
extern array<bool, N_SENSITIVITY_CHANNELS> sensitivity_channels;

//! Indices in Nuclide::reactions_ of the non-fission absorption reactions of
//! each nuclide, used to resolve capture pseudo-collisions by reaction
extern vector<vector<int>> sensitivity_capture_reactions;

} // namespace model

namespace simulation {

//! Ancestor chains of the neutrons in the source bank
extern vector<SensitivityChain> sensitivity_source_bank;

//! Ancestor chains of the sites in the fission bank
extern vector<SensitivityChain> sensitivity_fission_bank;

//! Ancestor chains of the sites sampled from the fission bank before they are
//! distributed to the source bank
extern vector<SensitivityChain> sensitivity_temp_bank;

//! Reference values of the delayed neutron fraction and ancestor lifetime
//! (in [s]) estimated during inactive generations. The scores of kinetics
//! parameter sensitivities are centered on these values to reduce variance.
extern array<double, 2> kinetics_reference;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

//! Check that a sensitivity calculation is compatible with other settings
void check_sensitivity_settings();

//! Determine which nuclides and reactions need to be recorded based on the
//! filters of the tallies requesting sensitivity scores
void initialize_sensitivity();

//! Allocate the sensitivity source and fission banks
void resize_sensitivity_banks();

//! Free memory associated with sensitivity calculations
void free_memory_sensitivity();

//! Record an event in the history of a neutron
//!
//! \param[in,out] p Particle
//! \param[in] i_nuclide Index of the nuclide in data::nuclides
//! \param[in] channel Perturbed channel (normalized MT for cross sections)
//! \param[in] E Energy associated with the event in [eV]
//! \param[in] sign +1 or -1
void record_sensitivity_event(
  Particle& p, int i_nuclide, int channel, double E, int sign);

//! Sample the events accounting for the derivative of the non-collision
//! probability along a flight path of the particle.
//!
//! The derivative of the probability of travelling a distance s without
//! collision with respect to a relative perturbation of a partial cross
//! section x is -Sigma_x * s. An unbiased estimate of this quantity is obtained
//! by sampling pseudo-collisions along the path with rate Sigma_t and scoring
//! -1 for the nuclide and reaction sampled at each pseudo-collision.
//!
//! \param[in,out] p Particle
//! \param[in] distance Distance travelled in the current material in [cm]
void sample_sensitivity_flight(Particle& p, double distance);

//! Store the ancestor chain of a fission site in the sensitivity fission bank
//!
//! \param[in] p Particle that created the fission site
//! \param[in] i_nuclide Index of the fissioned nuclide in data::nuclides
//! \param[in] site Fission site
//! \param[in] idx Index of the site in the fission bank
void bank_sensitivity_fission_site(
  const Particle& p, int i_nuclide, const SourceSite& site, int64_t idx);

//! Save the event log of a particle when it creates a secondary neutron
//!
//! \param[in,out] p Particle that created the secondary neutron
void save_sensitivity_secondary(Particle& p);

//! Restore the event log of a secondary neutron when it is revived
//!
//! \param[in,out] p Particle being revived from its local secondary bank
void restore_sensitivity_secondary(Particle& p);

//! Score the kinetics parameter tallies of a source neutron when it is born.
//! During inactive generations, the reference values of the kinetics
//! parameters are accumulated instead.
//!
//! \param[in] p Particle born from the source bank
void score_kinetics_sensitivity(const Particle& p);

//! Determine the reference values of the kinetics parameters from the
//! inactive generations. Called at the beginning of the first active batch.
void finalize_kinetics_reference();

//! Score the sensitivity numerator for the ancestor of the particle that is
//! ifp_n_generation generations old
//!
//! \param[in] p Particle
//! \param[in] i_tally Index of the tally in model::tallies
//! \param[in] score_index Index of the score in the tally results
void score_keff_sensitivity(const Particle& p, int i_tally, int score_index);

//==============================================================================
// Fission bank handling
//==============================================================================

//! Copy the chain of a fission site to the temporary bank
//!
//! \param[in] i_bank Index in the fission bank
//! \param[in] i_temp Index in the temporary bank
void copy_sensitivity_from_fission_bank(int64_t i_bank, int64_t i_temp);

//! Move a contiguous range of chains from the temporary bank to the source
//! bank. Chains are swapped so that allocated memory is reused.
//!
//! \param[in] i_temp Index of the first chain in the temporary bank
//! \param[in] n Number of chains to move
//! \param[in] i_bank Index of the first chain in the source bank
void move_sensitivity_to_source_bank(int64_t i_temp, int64_t n, int64_t i_bank);

#ifdef OPENMC_MPI

//! A contiguous range of sites transferred to or from another process
struct SensitivityTransfer {
  int neighbor;  //!< rank of the other process
  int64_t index; //!< index of the first site in the local bank
  int64_t n;     //!< number of sites
};

//! Exchange the chains of the sites that are sent between processes during
//! synchronization of the fission bank. Because chains have a variable length,
//! the number of events in each chain is communicated before the events.
//!
//! \param[in] sends Ranges of sites in the temporary bank sent by this process
//! \param[in] recvs Ranges of sites in the source bank received by this process
void exchange_sensitivity_chains(const vector<SensitivityTransfer>& sends,
  const vector<SensitivityTransfer>& recvs);

#endif

} // namespace openmc

#endif // OPENMC_SENSITIVITY_H
