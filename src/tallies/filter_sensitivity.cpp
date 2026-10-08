#include "openmc/tallies/filter_sensitivity.h"

#include <algorithm> // for find, upper_bound

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/container_util.h"
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
  channels_.clear();
  for (const auto& name : reactions) {
    if (contains(reactions_, name)) {
      throw std::runtime_error {fmt::format(
        "Duplicate reaction specified on sensitivity filter {}.", id())};
    }
    try {
      channels_.push_back(sensitivity_channels(name));
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error {fmt::format(
        "Invalid reaction '{}' specified on sensitivity filter {}: {}", name,
        id(), e.what())};
    }
    reactions_.push_back(name);
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

  // Build lookup tables from nuclide index and channel to filter position
  nuclide_bin_.assign(data::nuclide_map.size(), C_NONE);
  for (int i = 0; i < nuclides_.size(); ++i) {
    if (nuclide_bin_[nuclides_[i]] != C_NONE) {
      throw std::runtime_error {fmt::format(
        "Duplicate nuclide specified on sensitivity filter {}.", id())};
    }
    nuclide_bin_[nuclides_[i]] = i;
  }
  channel_positions_.assign(N_SENSITIVITY_CHANNELS, {});
  for (int i = 0; i < channels_.size(); ++i) {
    for (auto channel : channels_[i])
      channel_positions_[channel].push_back(i);
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

int SensitivityFilter::group(double E) const
{
  if (E < energies_.front() || E > energies_.back())
    return C_NONE;
  int n_groups = energies_.size() - 1;
  int group =
    std::upper_bound(energies_.begin(), energies_.end(), E) - energies_.begin();
  return std::min(group, n_groups) - 1;
}

void SensitivityFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  vector<std::string> nuclide_names;
  for (auto i_nuclide : nuclides_) {
    nuclide_names.push_back(data::nuclides[i_nuclide]->name_);
  }
  write_dataset(filter_group, "nuclides", nuclide_names);
  write_dataset(filter_group, "reactions", reactions_);
  write_dataset(filter_group, "energies", energies_);
}

std::string SensitivityFilter::text_label(int bin) const
{
  int n_groups = energies_.size() - 1;
  int group = bin % n_groups;
  int i_rx = (bin / n_groups) % reactions_.size();
  int i_nuc = bin / (n_groups * reactions_.size());
  return fmt::format("Sensitivity {} {} [{}, {})",
    data::nuclides[nuclides_[i_nuc]]->name_, reactions_[i_rx], energies_[group],
    energies_[group + 1]);
}

} // namespace openmc
