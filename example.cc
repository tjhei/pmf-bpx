// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2019 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------



// Test CEED bake-off problems BP1 and BP3 with Portable::MatrixFree

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

using namespace dealii;

// Mass Operator

template <int dim, int degree, typename Number, int n_q_points_1d>
class MassCellOperator
{
public:
  static const unsigned int n_q_points =
    dealii::Utilities::pow(n_q_points_1d, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, degree, n_q_points_1d, 1> fe(data);

    fe.read_dof_values(src);
    fe.evaluate(EvaluationFlags::values);

    data->for_each_quad_point([&](const int &q_point) {
      fe.submit_value(fe.get_value(q_point), q_point);
    });

    fe.integrate(EvaluationFlags::values);
    fe.distribute_local_to_global(dst);
  }
};

template <int dim,
          int degree,
          typename Number = double,
          typename VectorType =
            LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
          int n_q_points_1d = degree + 1>
class PortableMFMassOperator : public EnableObserverPointer
{
public:
  PortableMFMassOperator(const Portable::MatrixFree<dim, double> &data_in)
    : data(data_in)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = static_cast<Number>(0.);
    MassCellOperator<dim, degree, Number, n_q_points_1d> cell_operator;
    data.cell_loop(cell_operator, src, dst);

    // not required for BP1 as there are no hanging nodes or boundary
    // conditions: data.copy_constrained_values(src, dst);
  }

private:
  const Portable::MatrixFree<dim, Number> &data;
};



template <int dim, int degree, typename Number, int n_q_points_1d>
class LaplaceCellOperator
{
public:
  static const unsigned int n_q_points =
    dealii::Utilities::pow(n_q_points_1d, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, degree, n_q_points_1d, 1> fe(data, 0);

    fe.read_dof_values(src);
    fe.evaluate(EvaluationFlags::gradients);

    data->for_each_quad_point([&](const int &q_point) {
      fe.submit_gradient(fe.get_gradient(q_point), q_point);
    });

    fe.integrate(EvaluationFlags::gradients);
    fe.distribute_local_to_global(dst);
  }
};

template <int dim,
          int degree,
          typename Number = double,
          typename VectorType =
            LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
          int n_q_points_1d = degree + 1>
class PortableMFLaplaceOperator
{
public:
  PortableMFLaplaceOperator(const Portable::MatrixFree<dim, double> &data_in)
    : data(data_in)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = static_cast<Number>(0.);
    LaplaceCellOperator<dim, degree, Number, n_q_points_1d> cell_operator;
    data.cell_loop(cell_operator, src, dst);

    data.copy_constrained_values(src, dst);
  }

private:
  const Portable::MatrixFree<dim, Number> &data;
};


template <int dim, int degree, typename Number = double>
class Problem
{
public:
  Problem();

  void
  run();

  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

private:
  void
  setup_dofs();

  void
  solve();


  parallel::distributed::Triangulation<dim> tria;

  MappingQ<dim> mapping;

  FE_Q<dim> fe;

  DoFHandler<dim> dof;

  AffineConstraints<double> constraints;


  ConditionalOStream pcout;
};


template <int dim, int degree, typename Number>
Problem<dim, degree, Number>::Problem()
  : tria(MPI_COMM_WORLD)
  , mapping(1)
  , fe(degree)
  , dof(tria)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

template <int dim, int degree, typename Number>
void
Problem<dim, degree, Number>::setup_dofs()
{
  dof.distribute_dofs(fe);

  const IndexSet &owned_set    = dof.locally_owned_dofs();
  const IndexSet  relevant_set = DoFTools::extract_locally_relevant_dofs(dof);
  constraints.reinit(owned_set, relevant_set);
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values(dof,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
}



template <int dim, int degree, typename Number>
void
Problem<dim, degree, Number>::solve()
{
  if (true)
    {
      Portable::MatrixFree<dim, Number> mf_data;

      const QGauss<1> quad(degree + 1);
      typename Portable::MatrixFree<dim, Number>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_JxW_values;
      if (Kokkos::device_id() != -1)
        additional_data.team_size = 32;
      mf_data.reinit(mapping, dof, constraints, quad, additional_data);

      PortableMFMassOperator<dim, degree> mass_operator(mf_data);

      VectorType solution, rhs;

      mf_data.initialize_dof_vector(solution);
      mf_data.initialize_dof_vector(rhs);
      rhs = 1.0;

      for (unsigned int c = 0; c < 5; ++c)
        {
          Kokkos::Timer t;
          mass_operator.vmult(solution, rhs);
          const double time          = t.seconds();
          const double dofs_p_second = (double)solution.size() / time;
          pcout << "Mass operator (BP1) time: " << time
                << " MDoFs/s: " << dofs_p_second / 1e6 << std::endl;
        }
    }

  if (true)
    {
      Portable::MatrixFree<dim, Number> mf_data;

      const QGauss<1> quad(degree + 1);
      typename Portable::MatrixFree<dim, Number>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_gradients;
      if (Kokkos::device_id() != -1)
        additional_data.team_size = 32;
      mf_data.reinit(mapping, dof, constraints, quad, additional_data);

      PortableMFLaplaceOperator<dim, degree> laplace_operator(mf_data);

      VectorType solution, rhs;

      mf_data.initialize_dof_vector(solution);
      mf_data.initialize_dof_vector(rhs);

      for (unsigned int c = 0; c < 5; ++c)
        {
          Kokkos::Timer t;
          laplace_operator.vmult(solution, rhs);
          const double time          = t.seconds();
          const double dofs_p_second = (double)solution.size() / time;
          pcout << "Laplace operator (BP3) time: " << time
                << " MDoFs/s: " << dofs_p_second / 1e6 << std::endl;
        }
    }
}


template <int dim, int degree, typename Number>
void
Problem<dim, degree, Number>::run()
{
  //  pcout << std::setprecision(10);
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " MPI ranks and " << MultithreadInfo::n_threads() << " threads in "
#ifdef DEBUG
        << "DEBUG mode" << std::endl
#else
        << "RELEASE mode" << std::endl
#endif
        << "Kokkos device: " << Kokkos::device_id() << ' '
        << Kokkos::DefaultExecutionSpace().name() << std::endl
        << "dim: " << dim << std::endl
        << "Element: Q" << degree << std::endl;

  unsigned int n_refinements = 6;

  for (unsigned int i = 0; i < n_refinements; ++i)
    {
      if (i == 0)
        {
          GridGenerator::hyper_cube(tria);
          tria.refine_global(2);
        }
      else
        {
          tria.refine_global(1);
        }
      setup_dofs();

      pcout << "\nrefinement: " << i << ", n_dofs: " << dof.n_dofs()
            << std::endl;

      solve();
    }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const unsigned int           dim    = 3;
  const unsigned int           degree = 4;
  Problem<dim, degree, double> problem;
  problem.run();
}
