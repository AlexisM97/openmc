//! \file mesh_geometry.h
//! \brief Particle tracking on unstructured tetrahedral meshes

#ifndef OPENMC_MESH_GEOMETRY_H
#define OPENMC_MESH_GEOMETRY_H

#include <string>
#include <tuple>

#include "pugixml.hpp"

#include "openmc/array.h"
#include "openmc/cell.h"
#include "openmc/position.h"
#include "openmc/surface.h"
#include "openmc/universe.h"
#include "openmc/vector.h"

namespace openmc {

class MeshUniverse;

//==============================================================================
//! A cell of a mesh universe: the elements of one element block, which share
//! a material. Temperatures and density multipliers may be set per element.
//==============================================================================

class MeshCell : public Cell {
public:
  MeshCell(int32_t universe_index, int32_t block);

  bool contains(Position r, Direction u, int32_t on_surface) const override;

  std::pair<double, int32_t> distance(Position r, Direction u,
    int32_t on_surface, GeometryState* p) const override;

  BoundingBox bounding_box() const override;

  void to_hdf5_inner(hid_t group_id) const override;

  GeometryType geom_type() const override { return GeometryType::MESH; }

  //! Index of the element block of this cell in the mesh universe
  int32_t block() const { return block_; }

  const MeshUniverse& mesh_universe() const;

private:
  int32_t universe_index_; //!< Index of the mesh universe in model::universes
  int32_t block_;          //!< Element block of the cell
};

//==============================================================================
//! A surface of a mesh universe: either the set of all interior faces of the
//! mesh or a set of boundary faces, which may carry a boundary condition.
//! Geometric queries are answered by the mesh universe from the face that the
//! particle crosses.
//==============================================================================

class MeshSurface : public Surface {
public:
  MeshSurface(int32_t universe_index, int32_t boundary_set);

  double evaluate(Position r) const override;
  double distance(Position r, Direction u, bool coincident) const override;
  Direction normal(Position r) const override;
  Direction reflect(
    Position r, Direction u, GeometryState* p = nullptr) const override;
  void to_hdf5_inner(hid_t group_id) const override;
  GeometryType geom_type() const override { return GeometryType::MESH; }

  //! Boundary set of the surface (C_NONE for the interior faces)
  int32_t boundary_set() const { return boundary_set_; }

private:
  int32_t universe_index_;
  int32_t boundary_set_;
};

//==============================================================================
//! A universe whose geometry is a conforming mesh of linear tetrahedra. Each
//! tetrahedron belongs to an element of the original mesh (a hexahedron, for
//! instance, is split into tetrahedra), and each element to a block, which is
//! one cell of the universe. Particles are tracked from tetrahedron to
//! tetrahedron through the shared faces.
//==============================================================================

class MeshUniverse : public Universe {
public:
  explicit MeshUniverse(pugi::xml_node node);

  bool find_cell(GeometryState& p) const override;

  GeometryType geom_type() const override { return GeometryType::MESH; }

  void to_hdf5(hid_t group_id) const override;

  //! Tetrahedron containing a point
  //! \param[in] r Position in the local coordinates of the universe
  //! \param[in] u Direction, used to break ties on shared faces
  //! \return Index of the tetrahedron or C_NONE if outside of the mesh
  int32_t find_tet(Position r, Direction u) const;

  //! Distance to the exit face of a tetrahedron
  //! \param[in] tet Index of the tetrahedron
  //! \param[in] r Position inside (or on the boundary of) the tetrahedron
  //! \param[in] u Direction
  //! \param[in] entry_face Local face that the particle is on, or C_NONE
  //! \return Distance and local index of the exit face
  std::pair<double, int> exit_face(
    int32_t tet, Position r, Direction u, int entry_face) const;

  //! Distance to the exit face of the element of a tetrahedron, walking
  //! through the tetrahedra of the element
  //! \return Distance, tetrahedron and local face of the exit
  std::tuple<double, int32_t, int> walk_element(
    int32_t tet, Position r, Direction u, int entry_face) const;

  //! Find the tetrahedron of a particle off the faces (after a collision),
  //! searching its element first; updates the cell and properties if the
  //! element changes
  //! \return Whether the particle is in the mesh
  bool locate(GeometryState& p) const;

  //! Token of the surface of a local face of a tetrahedron
  int32_t face_surface(int32_t tet, int face) const;

  //! Move a particle through the face it has reached into the neighboring
  //! tetrahedron and set its cell, material, temperature and density
  void cross_face(GeometryState& p) const;

  //! Set the material, temperature and density multiplier of a particle in a
  //! tetrahedron, saving the previous values
  void set_properties(GeometryState& p) const;

  //! Outward unit normal of a local face of a tetrahedron
  Direction face_normal(int32_t tet, int face) const;

  //! Bounding box of the elements of a block
  BoundingBox block_bounding_box(int32_t block) const;

  //! Element of a tetrahedron
  int32_t element(int32_t tet) const { return tet_element_[tet]; }

  int32_t n_elements() const { return element_block_.size(); }
  int32_t n_tets() const { return tets_.size(); }
  int32_t block_of_element(int32_t e) const { return element_block_[e]; }
  int32_t cell_of_block(int32_t b) const { return block_cells_[b]; }
  int32_t block_id(int32_t b) const { return block_ids_[b]; }

  //! Throw if a temperature of an element is outside of the data range
  void check_temperature(int32_t e, double T) const;

  //! Set the temperature of an element (K)
  void set_element_temperature(int32_t e, double T);

  //! Temperature of an element (K)
  double element_temperature(int32_t e) const;

  //! Set the density multiplier of an element
  void set_element_density_mult(int32_t e, double mult);

  double element_density_mult(int32_t e) const;

  //! Volume of an element (cm^3)
  double element_volume(int32_t e) const { return element_volume_.at(e); }

  const std::string& filename() const { return filename_; }

private:
  void read_mesh(const std::string& filename, double length_multiplier);
  void build_faces();
  void build_search_grid();
  bool tet_contains(int32_t tet, Position r, double tol) const;

  std::string filename_;
  vector<Position> nodes_;
  vector<array<int32_t, 4>> tets_;
  vector<int32_t> tet_element_;
  vector<int32_t> element_block_;
  vector<int32_t> block_ids_;
  vector<int32_t> block_cells_; //!< Index in model::cells of each block

  //! Planes of the unique faces: unit normal and offset (n.r = d)
  vector<Direction> face_n_;
  vector<double> face_d_;
  //! Unique face of each local face of each tetrahedron (sign: + if the
  //! stored normal points out of the tetrahedron)
  vector<array<int32_t, 4>> tet_faces_;
  //! Neighbor through each local face, or -(1 + boundary surface index) if
  //! the face is on the boundary of the mesh
  vector<array<int32_t, 4>> neighbors_;
  //! Local face index of each face as seen from the neighbor
  vector<array<int8_t, 4>> neighbor_faces_;

  int32_t interior_surface_ {C_NONE}; //!< Index in model::surfaces
  vector<int32_t> boundary_surfaces_; //!< Index of each boundary set surface

  vector<int32_t> element_tet_start_; //!< Tetrahedra of each element (CSR)
  vector<int32_t> element_tets_;
  vector<double> element_volume_;       //!< cm^3
  vector<double> element_sqrtkT_;       //!< < 0: temperature of the cell
  vector<double> element_density_mult_; //!< < 0: multiplier of the cell

  // Uniform search grid for point location
  Position grid_lower_;
  Position grid_width_;
  array<int, 3> grid_n_;
  vector<int32_t> grid_start_;
  vector<int32_t> grid_tets_;
};

//==============================================================================
// Non-member functions
//==============================================================================

void read_mesh_universes(pugi::xml_node node);

//! Mesh universe of the lowest coordinate level of a particle
const MeshUniverse& mesh_universe(const GeometryState& p);

} // namespace openmc

#endif // OPENMC_MESH_GEOMETRY_H
