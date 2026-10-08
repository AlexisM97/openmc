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

  //! Determine the filter bin corresponding to a sensitivity event
  //!
  //! \param[in] event Event recorded in a neutron history
  //! \return Filter bin or C_NONE if the event is not binned by this filter
  int get_bin(const SensitivityEvent& event) const;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int>& nuclides() const { return nuclides_; }
  const vector<int>& reactions() const { return reactions_; }
  const vector<double>& energies() const { return energies_; }

  void set_nuclides(const vector<std::string>& nuclides);
  void set_reactions(const vector<std::string>& reactions);
  void set_energies(const vector<double>& energies);

private:
  //! Update the number of bins and lookup tables
  void update();

  //----------------------------------------------------------------------------
  // Data members

  vector<int> nuclides_;    //!< Indices in data::nuclides
  vector<int> reactions_;   //!< Perturbed reactions (SensitivityReaction)
  vector<double> energies_; //!< Energy group boundaries in [eV]

  vector<int> nuclide_bin_; //!< Position of each nuclide in nuclides_
  array<int, N_SENSITIVITY_REACTIONS> reaction_bin_; //!< Position in reactions_
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SENSITIVITY_H
