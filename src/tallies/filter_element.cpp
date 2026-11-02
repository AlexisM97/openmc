#include "openmc/tallies/filter_element.h"

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/mesh_geometry.h"
#include "openmc/xml_interface.h"

namespace openmc {

void ElementFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<int32_t>(node, "bins");
  if (bins.size() != 1) {
    fatal_error("Only one mesh universe can be specified per element filter.");
  }
  auto search = model::universe_map.find(bins[0]);
  if (search == model::universe_map.end()) {
    throw std::runtime_error {fmt::format(
      "Could not find universe {} specified on tally filter.", bins[0])};
  }
  this->set_universe(search->second);
}

void ElementFilter::set_universe(int32_t universe)
{
  auto* univ = dynamic_cast<MeshUniverse*>(model::universes[universe].get());
  if (!univ) {
    throw std::runtime_error {
      fmt::format("Universe {} of an element filter is not a mesh universe.",
        model::universes[universe]->id_)};
  }
  universe_ = universe;
  n_bins_ = univ->n_elements();
}

void ElementFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  if (p.lowest_coord().universe() != universe_ || p.mesh_tet() < 0)
    return;
  const auto& univ = mesh_universe(p);
  match.bins_.push_back(univ.element(p.mesh_tet()));
  match.weights_.push_back(1.0);
}

void ElementFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", model::universes[universe_]->id_);
}

std::string ElementFilter::text_label(int bin) const
{
  return fmt::format(
    "Element {} of mesh universe {}", bin, model::universes[universe_]->id_);
}

} // namespace openmc
