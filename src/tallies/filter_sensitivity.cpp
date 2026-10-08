#include "openmc/tallies/filter_sensitivity.h"

#include <algorithm> // for find, upper_bound

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/nuclide.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// SensitivityFilter implementation
//==============================================================================

void SensitivityFilter::from_xml(pugi::xml_node node)
{
  this->set_nuclides(get_node_array<std::string>(node, "nuclides"));
  this->set_reactions(get_node_array<std::string>(node, "reactions"));
  this->set_energies(get_node_array<double>(node, "energies"));
}

void SensitivityFilter::set_nuclides(const vector<std::string>& nuclides)
{
  nuclides_.clear();
  for (const auto& name : nuclides) {
    auto it = data::nuclide_map.find(name);
    if (it == data::nuclide_map.end()) {
      throw std::runtime_error {fmt::format(
        "Could not find nuclide {} specified on sensitivity filter {} in any "
        "material.",
        name, id())};
    }
    nuclides_.push_back(it->second);
  }
  this->update();
}

void SensitivityFilter::set_reactions(const vector<std::string>& reactions)
{
  reactions_.clear();
  for (const auto& name : reactions) {
    auto it = std::find(SENSITIVITY_REACTION_NAMES.begin(),
      SENSITIVITY_REACTION_NAMES.end(), name);
    if (it == SENSITIVITY_REACTION_NAMES.end()) {
      throw std::runtime_error {
        fmt::format("Unknown reaction '{}' specified on sensitivity filter {}.",
          name, id())};
    }
    reactions_.push_back(it - SENSITIVITY_REACTION_NAMES.begin());
  }
  this->update();
}

void SensitivityFilter::set_energies(const vector<double>& energies)
{
  for (int i = 1; i < energies.size(); ++i) {
    if (energies[i] <= energies[i - 1]) {
      throw std::runtime_error {
        "Energy bins must be monotonically increasing."};
    }
  }
  energies_ = energies;
  this->update();
}

void SensitivityFilter::update()
{
  int n_groups = energies_.empty() ? 0 : energies_.size() - 1;
  n_bins_ = nuclides_.size() * reactions_.size() * n_groups;

  // Build lookup tables from nuclide index and reaction to filter position
  nuclide_bin_.assign(data::nuclide_map.size(), C_NONE);
  for (int i = 0; i < nuclides_.size(); ++i) {
    if (nuclide_bin_[nuclides_[i]] != C_NONE) {
      throw std::runtime_error {fmt::format(
        "Duplicate nuclide specified on sensitivity filter {}.", id())};
    }
    nuclide_bin_[nuclides_[i]] = i;
  }
  reaction_bin_.fill(C_NONE);
  for (int i = 0; i < reactions_.size(); ++i) {
    if (reaction_bin_[reactions_[i]] != C_NONE) {
      throw std::runtime_error {fmt::format(
        "Duplicate reaction specified on sensitivity filter {}.", id())};
    }
    reaction_bin_[reactions_[i]] = i;
  }
}

void SensitivityFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // Note that the bin is set to zero here, but the actual bins are determined
  // by the events in the ancestry of the particle (tally_scoring.cpp)
  match.bins_.push_back(0);
  match.weights_.push_back(1.0);
}

int SensitivityFilter::get_bin(const SensitivityEvent& event) const
{
  int i_nuclide = event.nuclide();
  if (i_nuclide >= nuclide_bin_.size())
    return C_NONE;
  int i_nuc_bin = nuclide_bin_[i_nuclide];
  if (i_nuc_bin == C_NONE)
    return C_NONE;
  int i_rx_bin = reaction_bin_[event.reaction()];
  if (i_rx_bin == C_NONE)
    return C_NONE;

  // Determine energy group
  double E = event.E;
  if (E < energies_.front() || E > energies_.back())
    return C_NONE;
  int n_groups = energies_.size() - 1;
  int group =
    std::upper_bound(energies_.begin(), energies_.end(), E) - energies_.begin();
  group = std::min(group, n_groups) - 1;

  return (i_nuc_bin * reactions_.size() + i_rx_bin) * n_groups + group;
}

void SensitivityFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  vector<std::string> nuclide_names;
  for (auto i_nuclide : nuclides_) {
    nuclide_names.push_back(data::nuclides[i_nuclide]->name_);
  }
  vector<std::string> reaction_names;
  for (auto rx : reactions_) {
    reaction_names.push_back(SENSITIVITY_REACTION_NAMES[rx]);
  }
  write_dataset(filter_group, "nuclides", nuclide_names);
  write_dataset(filter_group, "reactions", reaction_names);
  write_dataset(filter_group, "energies", energies_);
}

std::string SensitivityFilter::text_label(int bin) const
{
  int n_groups = energies_.size() - 1;
  int group = bin % n_groups;
  int i_rx = (bin / n_groups) % reactions_.size();
  int i_nuc = bin / (n_groups * reactions_.size());
  return fmt::format("Sensitivity {} {} [{}, {})",
    data::nuclides[nuclides_[i_nuc]]->name_,
    SENSITIVITY_REACTION_NAMES[reactions_[i_rx]], energies_[group],
    energies_[group + 1]);
}

} // namespace openmc
