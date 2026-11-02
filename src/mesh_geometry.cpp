#include "openmc/mesh_geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <fmt/core.h>

#include "openmc/boundary_condition.h"
#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/nuclide.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/particle.h"
#include "openmc/settings.h"
#include "openmc/string_utils.h"
#include "openmc/xml_interface.h"

namespace openmc {

namespace {

//! Local node indices of the face opposite each node of a tetrahedron
constexpr int TET_FACE_NODES[4][3] = {
  {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

//! Unit normal of the face crossed last on this thread, for surface tallies
//! (MeshSurface::normal has no access to the particle)
thread_local Direction last_face_normal {0.0, 0.0, 1.0};

} // namespace

//==============================================================================
// MeshCell implementation
//==============================================================================

MeshCell::MeshCell(int32_t universe_index, int32_t block)
  : universe_index_ {universe_index}, block_ {block}
{
  fill_ = C_NONE;
}

const MeshUniverse& MeshCell::mesh_universe() const
{
  return static_cast<const MeshUniverse&>(*model::universes[universe_index_]);
}

bool MeshCell::contains(Position r, Direction u, int32_t on_surface) const
{
  const auto& univ = mesh_universe();
  int32_t tet = univ.find_tet(r, u);
  return tet != C_NONE &&
         univ.block_of_element(univ.element(tet)) == block_;
}

std::pair<double, int32_t> MeshCell::distance(
  Position r, Direction u, int32_t on_surface, GeometryState* p) const
{
  if (!p) {
    fatal_error("Distances in mesh cells require the particle state.");
  }
  const auto& univ = mesh_universe();
  int32_t& tet = p->mesh_tet();
  if (tet == C_NONE) {
    tet = univ.find_tet(r, u);
    p->mesh_face() = C_NONE;
    if (tet == C_NONE) {
      return {INFTY, -1};
    }
  }
  int entry = on_surface == SURFACE_NONE ? C_NONE : p->mesh_face();
  auto [d, face] = univ.exit_face(tet, r, u, entry);
  p->mesh_exit_face() = face;
  if (face == C_NONE) {
    return {INFTY, -1};
  }
  return {d, univ.face_surface(tet, face)};
}

BoundingBox MeshCell::bounding_box() const
{
  return mesh_universe().block_bounding_box(block_);
}

void MeshCell::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "geom_type", "mesh", false);
  write_dataset(group_id, "block", mesh_universe().block_id(block_));
}

//==============================================================================
// MeshSurface implementation
//==============================================================================

MeshSurface::MeshSurface(int32_t universe_index, int32_t boundary_set)
  : universe_index_ {universe_index}, boundary_set_ {boundary_set}
{}

double MeshSurface::evaluate(Position r) const
{
  fatal_error("Mesh surfaces cannot be evaluated at a point.");
}

double MeshSurface::distance(Position r, Direction u, bool coincident) const
{
  fatal_error("Distances to mesh surfaces are computed by the mesh cells.");
}

Direction MeshSurface::normal(Position r) const
{
  return last_face_normal;
}

Direction MeshSurface::reflect(Position r, Direction u, GeometryState* p) const
{
  if (!p) {
    fatal_error("Reflection on a mesh surface requires the particle state.");
  }
  const auto& univ =
    static_cast<const MeshUniverse&>(*model::universes[universe_index_]);
  Direction n = univ.face_normal(p->mesh_tet(), p->mesh_exit_face());
  last_face_normal = n;
  // The particle stays in its tetrahedron, on the reflecting face
  p->mesh_face() = p->mesh_exit_face();
  return u - 2.0 * u.dot(n) * n;
}

void MeshSurface::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "geom_type", "mesh", false);
  write_dataset(group_id, "boundary_set", boundary_set_);
  write_string(group_id, "boundary_type", bc_ ? bc_->type() : "transmission",
    false);
}

//==============================================================================
// MeshUniverse implementation
//==============================================================================

MeshUniverse::MeshUniverse(pugi::xml_node node)
{
  if (!check_for_node(node, "id")) {
    fatal_error("Must specify the id of a mesh universe.");
  }
  id_ = std::stoi(get_node_value(node, "id"));
  if (!check_for_node(node, "filename")) {
    fatal_error(
      fmt::format("Must specify the mesh file of mesh universe {}.", id_));
  }
  filename_ = get_node_value(node, "filename");
  if (!starts_with(filename_, "/")) {
    filename_ = dir_name(settings::path_input) + filename_;
  }
  if (!file_exists(filename_)) {
    fatal_error(fmt::format("Mesh file '{}' does not exist.", filename_));
  }
  double length_multiplier = 1.0;
  if (check_for_node(node, "length_multiplier")) {
    length_multiplier = std::stod(get_node_value(node, "length_multiplier"));
  }
  read_mesh(filename_, length_multiplier);

  int32_t univ_index = model::universes.size();

  // Cells: one per element block
  int32_t max_cell_id = 0;
  for (const auto& c : model::cells)
    max_cell_id = std::max(max_cell_id, c->id_);
  block_cells_.assign(block_ids_.size(), C_NONE);
  for (pugi::xml_node cell_node : node.children("cell")) {
    if (!check_for_node(cell_node, "block")) {
      fatal_error(fmt::format(
        "Cells of mesh universe {} must give their element block.", id_));
    }
    int32_t block_id = std::stoi(get_node_value(cell_node, "block"));
    auto it = std::find(block_ids_.begin(), block_ids_.end(), block_id);
    if (it == block_ids_.end()) {
      fatal_error(fmt::format(
        "Mesh universe {} has no element block {}.", id_, block_id));
    }
    int32_t b = it - block_ids_.begin();
    if (block_cells_[b] != C_NONE) {
      fatal_error(fmt::format(
        "Element block {} of mesh universe {} has two cells.", block_id, id_));
    }
    auto c = make_unique<MeshCell>(univ_index, b);
    c->id_ = check_for_node(cell_node, "id")
               ? std::stoi(get_node_value(cell_node, "id"))
               : ++max_cell_id;
    max_cell_id = std::max(max_cell_id, c->id_);
    c->universe_ = id_;
    if (check_for_node(cell_node, "name")) {
      c->name_ = get_node_value(cell_node, "name");
    }
    if (!check_for_node(cell_node, "material")) {
      fatal_error(fmt::format("Cell {} of mesh universe {} has no material.",
        c->id_, id_));
    }
    c->material_ = parse_cell_material_xml(cell_node, c->id_);
    if (c->material_.size() != 1) {
      fatal_error(fmt::format(
        "Cell {} of a mesh universe must have one material.", c->id_));
    }
    if (check_for_node(cell_node, "temperature")) {
      c->sqrtkT_ = parse_cell_temperature_xml(cell_node, c->id_);
      for (auto& T : c->sqrtkT_)
        T = std::sqrt(K_BOLTZMANN * T);
    }
    if (model::cell_map.count(c->id_)) {
      fatal_error(
        fmt::format("Two or more cells use the same unique ID: {}", c->id_));
    }
    block_cells_[b] = model::cells.size();
    model::cell_map[c->id_] = model::cells.size();
    model::cells.push_back(std::move(c));
  }
  for (int b = 0; b < block_ids_.size(); ++b) {
    if (block_cells_[b] == C_NONE) {
      fatal_error(fmt::format("Element block {} of mesh universe {} has no "
                              "cell.",
        block_ids_[b], id_));
    }
  }

  // Surfaces: the interior faces, the boundary faces without a set given in
  // the XML (default boundary condition) and one per boundary set
  int32_t max_surf_id = 0;
  for (const auto& s : model::surfaces)
    max_surf_id = std::max(max_surf_id, s->id_);
  auto add_surface = [&](int32_t set, const std::string& bc) {
    auto s = make_unique<MeshSurface>(univ_index, set);
    s->id_ = ++max_surf_id;
    std::string type = bc;
    to_lower(type);
    if (type == "vacuum") {
      s->bc_ = make_unique<VacuumBC>();
    } else if (type == "reflective" || type == "reflect" ||
               type == "reflecting") {
      s->bc_ = make_unique<ReflectiveBC>();
    } else if (type != "transmission" && type != "transmit") {
      fatal_error(fmt::format("Boundary condition '{}' is not supported on "
                              "mesh universe {} (vacuum, reflective or "
                              "transmission).",
        bc, id_));
    }
    model::surface_map[s->id_] = model::surfaces.size();
    model::surfaces.push_back(std::move(s));
    return static_cast<int32_t>(model::surfaces.size() - 1);
  };
  interior_surface_ = add_surface(C_NONE, "transmission");
  std::string default_bc = "vacuum";
  if (check_for_node(node, "boundary")) {
    default_bc = get_node_value(node, "boundary");
  }
  int32_t default_surface = add_surface(C_NONE, default_bc);
  std::unordered_map<int32_t, int32_t> set_surface;
  for (pugi::xml_node bnode : node.children("boundary")) {
    int32_t set = std::stoi(get_node_value(bnode, "set"));
    set_surface[set] = add_surface(set, get_node_value(bnode, "type"));
  }

  // Exterior faces: neighbor = -(1 + surface index)
  for (int32_t t = 0; t < tets_.size(); ++t) {
    for (int k = 0; k < 4; ++k) {
      if (neighbors_[t][k] >= 0)
        continue;
      int32_t set = -neighbors_[t][k] - 2; // see build_faces
      auto it = set_surface.find(set);
      int32_t s = it == set_surface.end() ? default_surface : it->second;
      neighbors_[t][k] = -(1 + s);
    }
  }

  element_volume_.assign(element_block_.size(), 0.0);
  for (int32_t t = 0; t < tets_.size(); ++t) {
    const auto& x = tets_[t];
    Position a = nodes_[x[0]];
    Position b = nodes_[x[1]] - a, c = nodes_[x[2]] - a, d = nodes_[x[3]] - a;
    element_volume_[tet_element_[t]] += std::abs(b.dot(c.cross(d))) / 6.0;
  }
  element_sqrtkT_.assign(element_block_.size(), -1.0);
  element_density_mult_.assign(element_block_.size(), -1.0);
  build_search_grid();
}

void MeshUniverse::read_mesh(const std::string& filename, double mult)
{
  hid_t file = file_open(filename, 'r');
  vector<double> xyz;
  vector<int32_t> tets, face_sets;
  read_dataset(file, "nodes", xyz);
  read_dataset(file, "tets", tets);
  read_dataset(file, "tet_element", tet_element_);
  read_dataset(file, "element_block", element_block_);
  read_dataset(file, "block_ids", block_ids_);
  if (object_exists(file, "tet_face_sets")) {
    read_dataset(file, "tet_face_sets", face_sets);
  }
  file_close(file);

  if (xyz.size() % 3 != 0 || tets.size() % 4 != 0 ||
      tet_element_.size() != tets.size() / 4) {
    fatal_error(fmt::format("Mesh file '{}' is inconsistent.", filename));
  }
  nodes_.resize(xyz.size() / 3);
  for (int i = 0; i < nodes_.size(); ++i) {
    nodes_[i] = mult * Position {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  }
  tets_.resize(tets.size() / 4);
  for (int t = 0; t < tets_.size(); ++t) {
    for (int k = 0; k < 4; ++k) {
      tets_[t][k] = tets[4 * t + k];
      if (tets_[t][k] < 0 || tets_[t][k] >= nodes_.size()) {
        fatal_error(fmt::format(
          "Tetrahedron {} of mesh file '{}' has an invalid node.", t,
          filename));
      }
    }
  }
  for (auto b : element_block_) {
    if (b < 0 || b >= block_ids_.size()) {
      fatal_error(
        fmt::format("Mesh file '{}' has an invalid element block.", filename));
    }
  }
  build_faces();

  // Boundary sets of the exterior faces, encoded as -(2 + set) until the
  // surfaces exist (-1: no set)
  for (int32_t t = 0; t < tets_.size(); ++t) {
    for (int k = 0; k < 4; ++k) {
      if (neighbors_[t][k] < 0) {
        int32_t set = face_sets.empty() ? -1 : face_sets[4 * t + k];
        neighbors_[t][k] = -(2 + set);
      }
    }
  }
}

void MeshUniverse::build_faces()
{
  int32_t n = tets_.size();
  // Sorted node triples of all local faces
  struct FaceKey {
    array<int32_t, 3> nodes;
    int32_t local; // 4 * tet + face
  };
  vector<FaceKey> keys(4 * n);
  for (int32_t t = 0; t < n; ++t) {
    for (int k = 0; k < 4; ++k) {
      array<int32_t, 3> f;
      for (int j = 0; j < 3; ++j)
        f[j] = tets_[t][TET_FACE_NODES[k][j]];
      std::sort(f.begin(), f.end());
      keys[4 * t + k] = {f, 4 * t + k};
    }
  }
  std::sort(keys.begin(), keys.end(),
    [](const FaceKey& a, const FaceKey& b) { return a.nodes < b.nodes; });

  tet_faces_.assign(n, {0, 0, 0, 0});
  neighbors_.assign(n, {-1, -1, -1, -1});
  neighbor_faces_.assign(n, {-1, -1, -1, -1});
  face_n_.clear();
  face_d_.clear();
  int n_degenerate = 0;
  for (int i = 0; i < keys.size();) {
    int j = i + 1;
    while (j < keys.size() && keys[j].nodes == keys[i].nodes)
      ++j;
    if (j - i > 2) {
      fatal_error(fmt::format("A face of mesh '{}' is shared by more than two "
                              "tetrahedra.",
        filename_));
    }
    // Plane of the face from its sorted nodes, so that both tetrahedra
    // sharing the face use exactly the same plane
    const auto& f = keys[i].nodes;
    Position a = nodes_[f[0]];
    Direction nrm = (nodes_[f[1]] - a).cross(nodes_[f[2]] - a);
    double norm = nrm.norm();
    if (norm == 0.0) {
      ++n_degenerate;
      norm = 1.0;
    }
    nrm /= norm;
    int32_t face = face_n_.size();
    face_n_.push_back(nrm);
    face_d_.push_back(nrm.dot(a));
    for (int m = i; m < j; ++m) {
      int32_t t = keys[m].local / 4;
      int k = keys[m].local % 4;
      // Outward if the opposite node is on the negative side
      double side = nrm.dot(nodes_[tets_[t][k]]) - face_d_.back();
      int sign = side < 0.0 ? 1 : -1;
      tet_faces_[t][k] = sign * (face + 1);
    }
    if (j - i == 2) {
      int32_t t0 = keys[i].local / 4, t1 = keys[i + 1].local / 4;
      int k0 = keys[i].local % 4, k1 = keys[i + 1].local % 4;
      neighbors_[t0][k0] = t1;
      neighbors_[t1][k1] = t0;
      neighbor_faces_[t0][k0] = k1;
      neighbor_faces_[t1][k1] = k0;
    }
    i = j;
  }
  if (n_degenerate > 0) {
    warning(fmt::format(
      "Mesh '{}' has {} degenerate faces.", filename_, n_degenerate));
  }
}

void MeshUniverse::build_search_grid()
{
  BoundingBox box = BoundingBox::inverted();
  for (const auto& x : nodes_)
    box.expand(x);
  // About two tetrahedra per grid cell
  Position size = box.max - box.min;
  double volume = std::max(size.x, 1e-12) * std::max(size.y, 1e-12) *
                  std::max(size.z, 1e-12);
  double h = std::cbrt(2.0 * volume / std::max<size_t>(tets_.size(), 1));
  for (int i = 0; i < 3; ++i) {
    grid_n_[i] = std::max(1, static_cast<int>(std::ceil(size[i] / h)));
  }
  grid_lower_ = box.min;
  grid_width_ = {std::max(size.x, 1e-12) / grid_n_[0],
    std::max(size.y, 1e-12) / grid_n_[1], std::max(size.z, 1e-12) / grid_n_[2]};

  auto index = [&](Position r, int i) {
    int j = static_cast<int>((r[i] - grid_lower_[i]) / grid_width_[i]);
    return std::clamp(j, 0, grid_n_[i] - 1);
  };
  // Count, then fill (compressed rows)
  int64_t n_cells = static_cast<int64_t>(grid_n_[0]) * grid_n_[1] * grid_n_[2];
  vector<int32_t> count(n_cells + 1, 0);
  auto for_each_cell = [&](int32_t t, auto&& f) {
    array<int, 3> lo, hi;
    for (int i = 0; i < 3; ++i) {
      double a = INFTY, b = -INFTY;
      for (int k = 0; k < 4; ++k) {
        a = std::min(a, nodes_[tets_[t][k]][i]);
        b = std::max(b, nodes_[tets_[t][k]][i]);
      }
      Position pa = grid_lower_, pb = grid_lower_;
      pa[i] = a;
      pb[i] = b;
      lo[i] = index(pa, i);
      hi[i] = index(pb, i);
    }
    for (int x = lo[0]; x <= hi[0]; ++x)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int z = lo[2]; z <= hi[2]; ++z)
          f((static_cast<int64_t>(z) * grid_n_[1] + y) * grid_n_[0] + x);
  };
  for (int32_t t = 0; t < tets_.size(); ++t) {
    for_each_cell(t, [&](int64_t c) { ++count[c + 1]; });
  }
  std::partial_sum(count.begin(), count.end(), count.begin());
  grid_start_ = count;
  grid_tets_.resize(count.back());
  for (int32_t t = 0; t < tets_.size(); ++t) {
    for_each_cell(t, [&](int64_t c) { grid_tets_[count[c]++] = t; });
  }
}

bool MeshUniverse::tet_contains(int32_t tet, Position r, double tol) const
{
  for (int k = 0; k < 4; ++k) {
    int32_t f = tet_faces_[tet][k];
    double s = f > 0 ? 1.0 : -1.0;
    int32_t i = std::abs(f) - 1;
    if (s * (face_n_[i].dot(r) - face_d_[i]) > tol)
      return false;
  }
  return true;
}

int32_t MeshUniverse::find_tet(Position r, Direction u) const
{
  for (int i = 0; i < 3; ++i) {
    double x = (r[i] - grid_lower_[i]) / grid_width_[i];
    if (x < -1e-9 * grid_n_[i] || x > grid_n_[i] * (1.0 + 1e-9))
      return C_NONE;
  }
  int64_t c = 0;
  {
    array<int, 3> j;
    for (int i = 0; i < 3; ++i) {
      j[i] = std::clamp(
        static_cast<int>((r[i] - grid_lower_[i]) / grid_width_[i]), 0,
        grid_n_[i] - 1);
    }
    c = (static_cast<int64_t>(j[2]) * grid_n_[1] + j[1]) * grid_n_[0] + j[0];
  }
  // A point on a shared face belongs to the tetrahedron that the direction
  // points into; points outside by rounding are accepted with a tolerance
  Position r_ahead = r + TINY_BIT * u;
  for (int32_t i = grid_start_[c]; i < grid_start_[c + 1]; ++i) {
    if (tet_contains(grid_tets_[i], r_ahead, 0.0))
      return grid_tets_[i];
  }
  for (int32_t i = grid_start_[c]; i < grid_start_[c + 1]; ++i) {
    if (tet_contains(grid_tets_[i], r, FP_PRECISION))
      return grid_tets_[i];
  }
  return C_NONE;
}

std::pair<double, int> MeshUniverse::exit_face(
  int32_t tet, Position r, Direction u, int entry_face) const
{
  double d_min = INFTY;
  int face = C_NONE;
  for (int k = 0; k < 4; ++k) {
    if (k == entry_face)
      continue;
    int32_t f = tet_faces_[tet][k];
    double s = f > 0 ? 1.0 : -1.0;
    int32_t i = std::abs(f) - 1;
    double cos = s * face_n_[i].dot(u);
    if (cos <= 0.0)
      continue;
    double d = s * (face_d_[i] - face_n_[i].dot(r)) / cos;
    if (d < d_min) {
      d_min = d;
      face = k;
    }
  }
  return {std::max(d_min, 0.0), face};
}

int32_t MeshUniverse::face_surface(int32_t tet, int face) const
{
  int32_t nb = neighbors_[tet][face];
  int32_t s = nb >= 0 ? interior_surface_ : -nb - 1;
  return s + 1;
}

Direction MeshUniverse::face_normal(int32_t tet, int face) const
{
  int32_t f = tet_faces_[tet][face];
  return (f > 0 ? 1.0 : -1.0) * face_n_[std::abs(f) - 1];
}

void MeshUniverse::cross_face(GeometryState& p) const
{
  int32_t tet = p.mesh_tet();
  int face = p.mesh_exit_face();
  last_face_normal = face_normal(tet, face);
  int32_t nb = neighbors_[tet][face];
  if (nb < 0) {
    p.mark_as_lost(fmt::format("Particle {} left mesh universe {} through a "
                               "boundary without a boundary condition.",
      p.id(), id_));
    return;
  }
  p.mesh_face() = neighbor_faces_[tet][face];
  p.mesh_tet() = nb;
  p.lowest_coord().cell() = block_cells_[element_block_[tet_element_[nb]]];
  p.cell_instance() = 0;
  set_properties(p);
}

void MeshUniverse::set_properties(GeometryState& p) const
{
  const Cell& c = *model::cells[p.lowest_coord().cell()];
  int32_t e = tet_element_[p.mesh_tet()];
  p.material_last() = p.material();
  p.material() = c.material(0);
  p.sqrtkT_last() = p.sqrtkT();
  p.sqrtkT() = element_sqrtkT_[e] >= 0.0 ? element_sqrtkT_[e] : c.sqrtkT(0);
  p.density_mult_last() = p.density_mult();
  p.density_mult() = element_density_mult_[e] >= 0.0 ? element_density_mult_[e]
                                                     : c.density_mult(0);
}

bool MeshUniverse::find_cell(GeometryState& p) const
{
  int32_t tet = find_tet(p.r_local(), p.u_local());
  if (tet == C_NONE)
    return false;
  p.mesh_tet() = tet;
  p.mesh_face() = C_NONE;
  p.lowest_coord().cell() = block_cells_[element_block_[tet_element_[tet]]];
  return true;
}

BoundingBox MeshUniverse::block_bounding_box(int32_t block) const
{
  BoundingBox box = BoundingBox::inverted();
  for (int32_t t = 0; t < tets_.size(); ++t) {
    if (element_block_[tet_element_[t]] != block)
      continue;
    for (int k = 0; k < 4; ++k)
      box.expand(nodes_[tets_[t][k]]);
  }
  return box;
}

void MeshUniverse::check_temperature(int32_t e, double T) const
{
  if (settings::temperature_method == TemperatureMethod::INTERPOLATION &&
      (T < data::temperature_min - settings::temperature_tolerance ||
        T > data::temperature_max + settings::temperature_tolerance)) {
    throw std::runtime_error {
      fmt::format("Temperature of {} K of element {} of mesh universe {} is "
                  "outside of the range of the nuclear data ({}-{} K).",
        T, e, id_, data::temperature_min, data::temperature_max)};
  }
}

void MeshUniverse::set_element_temperature(int32_t e, double T)
{
  check_temperature(e, T);
  element_sqrtkT_.at(e) = std::sqrt(K_BOLTZMANN * T);
}

double MeshUniverse::element_temperature(int32_t e) const
{
  double s = element_sqrtkT_.at(e);
  if (s < 0.0)
    s = model::cells[block_cells_[element_block_[e]]]->sqrtkT(0);
  return s * s / K_BOLTZMANN;
}

void MeshUniverse::set_element_density_mult(int32_t e, double mult)
{
  if (mult < 0.0) {
    throw std::runtime_error {"Density multipliers must be non-negative."};
  }
  element_density_mult_.at(e) = mult;
}

double MeshUniverse::element_density_mult(int32_t e) const
{
  double m = element_density_mult_.at(e);
  return m >= 0.0
           ? m
           : model::cells[block_cells_[element_block_[e]]]->density_mult(0);
}

void MeshUniverse::to_hdf5(hid_t universes_group) const
{
  auto group = create_group(universes_group, fmt::format("universe {}", id_));
  write_string(group, "geom_type", "mesh", false);
  write_string(group, "filename", filename_, false);
  vector<int32_t> cell_ids;
  for (auto i : block_cells_)
    cell_ids.push_back(model::cells[i]->id_);
  write_dataset(group, "cells", cell_ids);
  write_dataset(group, "block_ids", block_ids_);
  write_attribute(group, "n_elements", n_elements());
  write_attribute(group, "n_tets", n_tets());
  close_group(group);
}

//==============================================================================
// Non-member functions
//==============================================================================

void read_mesh_universes(pugi::xml_node node)
{
  for (pugi::xml_node mesh_node : node.children("mesh_universe")) {
    auto univ = make_unique<MeshUniverse>(mesh_node);
    if (model::universe_map.count(univ->id_)) {
      fatal_error(fmt::format(
        "Two or more universes use the same unique ID: {}", univ->id_));
    }
    model::universe_map[univ->id_] = model::universes.size();
    model::universes.push_back(std::move(univ));
  }
}

const MeshUniverse& mesh_universe(const GeometryState& p)
{
  return static_cast<const MeshUniverse&>(
    *model::universes[p.lowest_coord().universe()]);
}

//==============================================================================
// C API
//==============================================================================

namespace {

//! Mesh universe of an ID, or nullptr with the error message set
MeshUniverse* find_mesh_universe(int32_t id)
{
  auto it = model::universe_map.find(id);
  if (it == model::universe_map.end()) {
    set_errmsg(fmt::format("No universe exists with ID={}.", id));
    return nullptr;
  }
  auto* univ = dynamic_cast<MeshUniverse*>(model::universes[it->second].get());
  if (!univ) {
    set_errmsg(fmt::format("Universe {} is not a mesh universe.", id));
  }
  return univ;
}

//! Apply f(i) to the n elements of a mesh universe of an ID
template<typename F>
int for_elements(int32_t id, int32_t n, F&& f)
{
  auto* univ = find_mesh_universe(id);
  if (!univ)
    return OPENMC_E_INVALID_ID;
  if (n != univ->n_elements()) {
    set_errmsg(fmt::format("Mesh universe {} has {} elements, not {}.", id,
      univ->n_elements(), n));
    return OPENMC_E_INVALID_SIZE;
  }
  try {
    for (int32_t e = 0; e < n; ++e)
      f(*univ, e);
  } catch (const std::exception& err) {
    set_errmsg(err.what());
    return OPENMC_E_INVALID_ARGUMENT;
  }
  return 0;
}

} // namespace

extern "C" int openmc_mesh_universe_get_n_elements(int32_t id, int32_t* n)
{
  auto* univ = find_mesh_universe(id);
  if (!univ)
    return OPENMC_E_INVALID_ID;
  *n = univ->n_elements();
  return 0;
}

extern "C" int openmc_mesh_universe_get_blocks(
  int32_t id, int32_t* block_ids, int32_t n)
{
  return for_elements(id, n, [&](const MeshUniverse& u, int32_t e) {
    block_ids[e] = u.block_id(u.block_of_element(e));
  });
}

extern "C" int openmc_mesh_universe_get_volumes(
  int32_t id, double* volumes, int32_t n)
{
  return for_elements(id, n, [&](const MeshUniverse& u, int32_t e) {
    volumes[e] = u.element_volume(e);
  });
}

extern "C" int openmc_mesh_universe_get_temperatures(
  int32_t id, double* T, int32_t n)
{
  return for_elements(id, n, [&](const MeshUniverse& u, int32_t e) {
    T[e] = u.element_temperature(e);
  });
}

extern "C" int openmc_mesh_universe_set_temperatures(
  int32_t id, const double* T, int32_t n)
{
  // Check all temperatures before setting any
  int err = for_elements(id, n,
    [&](const MeshUniverse& u, int32_t e) { u.check_temperature(e, T[e]); });
  if (err)
    return err;
  return for_elements(id, n,
    [&](MeshUniverse& u, int32_t e) { u.set_element_temperature(e, T[e]); });
}

extern "C" int openmc_mesh_universe_get_density_mults(
  int32_t id, double* mult, int32_t n)
{
  return for_elements(id, n, [&](const MeshUniverse& u, int32_t e) {
    mult[e] = u.element_density_mult(e);
  });
}

extern "C" int openmc_mesh_universe_set_density_mults(
  int32_t id, const double* mult, int32_t n)
{
  return for_elements(id, n, [&](MeshUniverse& u, int32_t e) {
    u.set_element_density_mult(e, mult[e]);
  });
}

} // namespace openmc
