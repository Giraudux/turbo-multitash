/**
 * Branch and bound algorithm to find the minimum of continuous binary functions
 * using interval arithmetic.
 *
 * OpenMP/MPI version
 *
 * Authors: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
 *          Alexis Giraudet <Alexis.Giraudet@etu.univ-nantes.fr>
 *          Dennis Bordet <Dennis.Bordet@etu.univ-nantes.fr>
 *
 * v. 3.0, 2016-04-08
 */

#define TM_EXTRA_DEPTH 0

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>

#include <mpi.h>
#include <syslog.h>

#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;

/**
 * Split a 2D box into four subboxes by splitting each dimension into two equal
 * subparts
 */
void tm_split_box(const interval &x, const interval &y, interval &xl,
                  interval &xr, interval &yl, interval &yr) {
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(), xm);
  xr = interval(xm, x.right());
  yl = interval(y.left(), ym);
  yr = interval(ym, y.right());
}

/**
 * Branch-and-bound minimization algorithm
 * \param itvfun Function to minimize
 * \param x Current bounds for 1st dimension
 * \param y Current bounds for 2nd dimension
 * \param threshold Threshold at which we should stop splitting
 * \param min_ub Current minimum upper bound
 * \param ml List of current minimizers
 */
void tm_minimize(itvfun f, const interval &x, const interval &y,
                 double threshold, double &min_ub,
                 minimizer_list &ml) // List of current minimizers
{
  interval fxy = f(x, y);

  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
// Discarding all saved boxes whose minimum lower bound is
// greater than the new minimum upper bound
#pragma omp critical
    {
      auto discard_begin = ml.lower_bound(minimizer{0, 0, min_ub, 0});
      ml.erase(discard_begin, ml.end());
    }
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) {
// We have potentially a new minimizer
#pragma omp critical
    { ml.insert(minimizer{x, y, fxy.left(), fxy.right()}); }
    return;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  tm_split_box(x, y, xl, xr, yl, yr);

#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
    tm_minimize(f, xl, yl, threshold, min_ub, ml);

#pragma omp section
    tm_minimize(f, xl, yr, threshold, min_ub, ml);

#pragma omp section
    tm_minimize(f, xr, yl, threshold, min_ub, ml);

#pragma omp section
    tm_minimize(f, xr, yr, threshold, min_ub, ml);
  }
}

/**
 * Read function and precision parameters on standard input
 */
void tm_read_fun_precision(opt_fun_t &fun, double &precision) {
  cout.precision(16);

  // Name of the function to optimize
  string choice_fun;

  bool good_choice;

  // Asking the user for the name of the function to optimize
  do {
    good_choice = true;

    cout << "Which function to optimize?\n";
    cout << "Possible choices: ";
    for (auto fname : functions) {
      cout << fname.first << " ";
    }
    cout << endl;
    cin >> choice_fun;

    try {
      fun = functions.at(choice_fun);
    } catch (out_of_range) {
      cerr << "Bad choice" << endl;
      good_choice = false;
    }
  } while (!good_choice);

  // Asking for the threshold below which a box is not split further
  cout << "Precision? ";
  cin >> precision;
}

/**
 * Return the box depth
 */
int tm_depth(int numprocs) {
  return int(ceil(log(double(numprocs)) / log(double(4.0)))) +
         int(TM_EXTRA_DEPTH);
}

/**
 * Return the number of boxes
 */
int tm_boxes(int numprocs) {
  return int(pow(double(4.0), double(tm_depth(numprocs))));
}

/**
 * Find the given box
 */
void tm_box(int box, int numprocs, const interval &x, const interval &y,
            interval &box_x, interval &box_y) {
  interval xl, xr, yl, yr;
  int boxes, depth;

  boxes = tm_boxes(numprocs);
  box_x = x;
  box_y = y;
  for (depth = tm_depth(numprocs); depth > 0; depth--) {
    tm_split_box(box_x, box_y, xl, xr, yl, yr);
    boxes /= 4;
    if (box < boxes) {
      box_x = xl;
      box_y = yl;
    } else if (box < 2 * boxes) {
      box_x = xr;
      box_y = yl;
    } else if (box < 3 * boxes) {
      box_x = xl;
      box_y = yr;
    } else {
      box_x = xr;
      box_y = yr;
    }
    box %= boxes;
  }
}

/**
 *
 */
int main(int argc, char *argv[]) {
  int numprocs, rank, status;
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub, local_min_ub;
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;
  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  char buff_bcast[sizeof(opt_fun_t) + sizeof(double)];
  int box, boxes;
  interval box_x, box_y;

  openlog(NULL, LOG_CONS | LOG_PID, LOG_USER);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  min_ub = local_min_ub = numeric_limits<double>::infinity();
  boxes = tm_boxes(numprocs);

  syslog(LOG_INFO, "depth = %d", tm_depth(numprocs));
  syslog(LOG_INFO, "boxes = %d", tm_boxes(numprocs));

  if (rank == 0) {
    tm_read_fun_precision(fun, precision);

    memcpy(buff_bcast, &fun, sizeof(fun));
    memcpy(buff_bcast + sizeof(fun), &precision, sizeof(precision));
  }

  auto start = chrono::high_resolution_clock::now();

  status =
      MPI_Bcast(buff_bcast, sizeof(buff_bcast), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Error_string(status, error_str, &error_len);
  syslog(LOG_INFO, "MPI_Bcast: %s", error_str);

  if (rank != 0) {
    memcpy(&fun, buff_bcast, sizeof(fun));
    memcpy(&precision, buff_bcast + sizeof(fun), sizeof(precision));
  }

  for (box = rank; box < boxes; box += numprocs) {
    tm_box(box, numprocs, fun.x, fun.y, box_x, box_y);
    syslog(LOG_INFO,
           "box = %d, x_left = %f, x_right = %f, y_left = %f, y_right = %f",
           box, box_x.left(), box_x.right(), box_y.left(), box_y.right());
    tm_minimize(fun.f, box_x, box_y, precision, local_min_ub, minimums);
  }

  status = MPI_Reduce(&local_min_ub, &min_ub, 1, MPI_DOUBLE, MPI_MIN, 0,
                      MPI_COMM_WORLD);
  MPI_Error_string(status, error_str, &error_len);
  syslog(LOG_INFO, "MPI_Reduce: %s", error_str);

  auto end = chrono::high_resolution_clock::now();

  syslog(LOG_INFO, "local_min_ub = %f", local_min_ub);

  if (rank == 0) {
    // Displaying all potential minimizers
    copy(minimums.begin(), minimums.end(),
         ostream_iterator<minimizer>(cout, "\n"));
    cout << "Number of minimizers: " << minimums.size() << endl;
    cout << "Upper bound for minimum: " << min_ub << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
  }

  MPI_Finalize();
  closelog();

  return EXIT_SUCCESS;
}
