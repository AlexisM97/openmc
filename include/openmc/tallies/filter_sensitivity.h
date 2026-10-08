#ifndef OPENMC_TALLIES_FILTER_SENSITIVITY_H
#define OPENMC_TALLIES_FILTER_SENSITIVITY_H

#include <string>

#include "openmc/sensitivity.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins sensitivity coefficients by perturbed nuclide, perturbed nuclear data,
//! and energy group.
//!
//! Bins are ordered with the nuclide varying slowest and the energy group
//! varying fastest. The get_all_bins functionality is not actually used; the
//! bins are determined from the events recorded in a neutron's ancestry in the
//! scoring subroutines.
//==============================================================================

class SensitivityFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~SensitivityFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "sensitivity"; }
  FilterType type() const override { return FilterType::SENSITIVITY; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //! Call a function for each filter bin corresponding to a sensitivity
  //! event. An event may correspond to several bins when the perturbed
  //! nuclear data overlap, e.g., 'inelastic' and '(n,2n)'.
  //!
  //! \param[in] event Event recorded in a neutron history
  //! \param[in] f Function called with each bin
  template<typename F>
  void for_each_bin(const SensitivityEvent& event, F&& f) const
  {
    int i_nuclide = event.nuclide();
    if (i_nuclide >= nuclide_bin_.size())
      return;
    int i_nuc_bin = nuclide_bin_[i_nuclide];
    if (i_nuc_bin == C_NONE)
      return;
    const auto& positions = channel_positions_[event.channel()];
    if (positions.empty())
      return;
    int group = this->group(event.E);
    if (group == C_NONE)
      return;
    int n_groups = energies_.size() - 1;
    for (auto i_rx : positions) {
      f((i_nuc_bin * reactions_.size() + i_rx) * n_groups + group);
    }
  }

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int>& nuclides() const { return nuclides_; }
  const vector<std::string>& reactions() const { return reactions_; }
  const vector<vector<int>>& channels() const { return channels_; }
  const vector<double>& energies() const { return energies_; }

  void set_nuclides(const vector<std::string>& nuclides);
  void set_reactions(const vector<std::string>& reactions);
  void set_energies(const vector<double>& energies);

private:
  //! Update the number of bins and lookup tables
  void update();

  //! Determine the energy group of an energy
  //!
  //! \param[in] E Energy in [eV]
  //! \return Energy group or C_NONE if outside of the group structure
  int group(double E) const;

  //----------------------------------------------------------------------------
  // Data members

  vector<int> nuclides_;          //!< Indices in data::nuclides
  vector<std::string> reactions_; //!< Names of perturbed nuclear data
  vector<vector<int>> channels_;  //!< Channels of each perturbed data
  vector<double> energies_;       //!< Energy group boundaries in [eV]

  vector<int> nuclide_bin_; //!< Position of each nuclide in nuclides_
  //! Positions in reactions_ that include each channel
  vector<vector<int>> channel_positions_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SENSITIVITY_H
