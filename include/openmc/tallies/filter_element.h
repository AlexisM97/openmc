#ifndef OPENMC_TALLIES_FILTER_ELEMENT_H
#define OPENMC_TALLIES_FILTER_ELEMENT_H

#include <cstdint>

#include "openmc/tallies/filter.h"

namespace openmc {

//==============================================================================
//! Bins events by the element of a mesh universe in which they occur. Track
//! lengths are exact, since tracks end on the faces of the elements.
//==============================================================================

class ElementFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~ElementFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "element"; }
  FilterType type() const override { return FilterType::ELEMENT; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  //! Set the mesh universe by its index in model::universes
  void set_universe(int32_t universe);

  int32_t universe() const { return universe_; }

private:
  //! Index of the mesh universe in model::universes
  int32_t universe_ {-1};
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_ELEMENT_H
