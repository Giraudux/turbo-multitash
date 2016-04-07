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
 * v. 4.0, 2016-04-08
 */

#define EXTRA_DEPTH 0

#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <mpi.h>
#include <syslog.h>

#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;

enum tm_tag_e { TM_TAG_MIN_UB, TM_TAG_BOX };

//
int tm_current_box;
int tm_numprocs;
int tm_rank;
double tm_min_ub;
minimizer_list tm_minimums;
mutex tm_minimums_mtx;
mutex tm_min_ub_mtx;
condition_variable tm_min_ub_cv;

// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval &x, const interval &y, interval &xl, interval &xr,
               interval &yl, interval &yr) {
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(), xm);
  xr = interval(xm, x.right());
  yl = interval(y.left(), ym);
  yr = interval(ym, y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,           // Function to minimize
              const interval &x,  // Current bounds for 1st dimension
              const interval &y,  // Current bounds for 2nd dimension
              double threshold,   // Threshold at which we should stop splitting
              double &min_ub,     // Current minimum upper bound
              minimizer_list &ml) // List of current minimizers
{
  interval fxy = f(x, y);

  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    // unique_lock<mutex> ulock(tm_min_ub_mtx);
    min_ub = fxy.right();
    tm_min_ub_cv.notify_all();
    // Discarding all saved boxes whose minimum lower bound is
    // greater than the new minimum upper bound
    //#pragma omp critical
    {
      tm_minimums_mtx.lock();
      auto discard_begin = ml.lower_bound(minimizer{0, 0, min_ub, 0});
      ml.erase(discard_begin, ml.end());
      tm_minimums_mtx.unlock();
    }
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) {
    // We have potentially a new minimizer
    //#pragma omp critical
    {
      tm_minimums_mtx.lock();
      ml.insert(minimizer{x, y, fxy.left(), fxy.right()});
      tm_minimums_mtx.unlock();
    }
    return;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x, y, xl, xr, yl, yr);

#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
    minimize(f, xl, yl, threshold, min_ub, ml);

#pragma omp section
    minimize(f, xl, yr, threshold, min_ub, ml);

#pragma omp section
    minimize(f, xr, yl, threshold, min_ub, ml);

#pragma omp section
    minimize(f, xr, yr, threshold, min_ub, ml);
  }
}

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

int tm_max_boxes(int numprocs) {
  int depth =
      int(ceil(log(double(numprocs)) / log(double(4.0)))) + int(EXTRA_DEPTH);
  int boxes = int(pow(double(4.0), double(depth - 1)));
  return boxes;
}

void tm_find_box(int rank, int numprocs, const interval &x, const interval &y,
                 interval &box_x, interval &box_y) {
  interval xl, xr, yl, yr;
  int boxes, depth;

  box_x = x;
  box_y = y;
  for (depth = int(ceil(log(double(numprocs)) / log(double(4.0)))) +
               int(EXTRA_DEPTH);
       depth > 0; depth--) {
    split_box(box_x, box_y, xl, xr, yl, yr);
    boxes = int(pow(double(4.0), double(depth - 1)));
    if (rank < boxes) {
      box_x = xl;
      box_y = yl;
    } else if (rank < 2 * boxes) {
      box_x = xr;
      box_y = yl;
    } else if (rank < 3 * boxes) {
      box_x = xl;
      box_y = yr;
    } else {
      box_x = xr;
      box_y = yr;
    }
    rank = rank % boxes;
  }
}

void tm_box_provider() {
  int status;
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  MPI_Status status_mpi;
  int max_boxes;
  int box;

  syslog(LOG_INFO, "tm_box_provider: start");

  max_boxes = tm_max_boxes(tm_numprocs);

  for (;;) {
    status = MPI_Recv(&box, 1, MPI_INT, MPI_ANY_SOURCE, TM_TAG_BOX,
                      MPI_COMM_WORLD, &status_mpi);
    MPI_Error_string(status, error_str, &error_len);
    syslog(LOG_INFO, "tm_box_provider: MPI_Recv: %s", error_str);

    if (status == MPI_SUCCESS) {
      if (tm_current_box < max_boxes) {
        box = tm_current_box;
        tm_current_box++;
      } else {
        box = -1;
      }

      status = MPI_Send(&box, 1, MPI_INT, status_mpi.MPI_SOURCE, TM_TAG_BOX,
                        MPI_COMM_WORLD);
      MPI_Error_string(status, error_str, &error_len);
      syslog(LOG_INFO, "tm_box_provider: MPI_Send: %s", error_str);
    }
  }
}

void tm_min_ub_receiver() {
  int status;
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  MPI_Status status_mpi;
  double new_min_ub;

  syslog(LOG_INFO, "tm_min_ub_receiver: start");

  for (;;) {
    if (tm_rank == 0) {
      status = MPI_Recv(&new_min_ub, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                        TM_TAG_MIN_UB, MPI_COMM_WORLD, &status_mpi);
      MPI_Error_string(status, error_str, &error_len);
      syslog(LOG_INFO, "tm_min_ub_receiver: MPI_Recv: %s", error_str);

      if (status != MPI_SUCCESS) {
        continue;
      }
    }

    status = MPI_Bcast(&new_min_ub, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Error_string(status, error_str, &error_len);
    syslog(LOG_INFO, "tm_min_ub_receiver: MPI_Bcast: %s", error_str);

    if (status == MPI_SUCCESS && new_min_ub < tm_min_ub) {
      tm_min_ub = new_min_ub;
      tm_minimums_mtx.lock();
      auto discard_begin =
          tm_minimums.lower_bound(minimizer{0, 0, new_min_ub, 0});
      tm_minimums.erase(discard_begin, tm_minimums.end());
      tm_minimums_mtx.unlock();
    }
  }
}

void tm_min_ub_sender() {
  int status;
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  double new_min_ub;
  unique_lock<mutex> ulock(tm_min_ub_mtx);

  syslog(LOG_INFO, "tm_min_ub_sender: start");

  for (;;) {
    tm_min_ub_cv.wait(ulock);

    new_min_ub = tm_min_ub;
    status =
        MPI_Send(&new_min_ub, 1, MPI_DOUBLE, 0, TM_TAG_MIN_UB, MPI_COMM_WORLD);
    MPI_Error_string(status, error_str, &error_len);
    syslog(LOG_INFO, "tm_min_ub_sender: MPI_Send: %s", error_str);
  }
}

int main(int argc, char *argv[]) {
  int status;
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  MPI_Status status_mpi;
  char buff[sizeof(opt_fun_t) + sizeof(double)];
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub;
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is
  // greater than the new current upper bound
  // Threshold at which we should stop splitting a box
  double precision;
  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  int box;
  interval box_x, box_y;

  openlog(NULL, LOG_CONS | LOG_PID, LOG_USER);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &tm_numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &tm_rank);

  tm_current_box = tm_numprocs;
  tm_min_ub = numeric_limits<double>::infinity();

  if (tm_rank == 0) {
    thread box_provider(tm_box_provider);
    box_provider.detach();

    tm_read_fun_precision(fun, precision);
    memcpy(buff, &fun, sizeof(fun));
    memcpy(buff + sizeof(fun), &precision, sizeof(precision));
  }

  status = MPI_Bcast(buff, sizeof(buff), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Error_string(status, error_str, &error_len);
  syslog(LOG_INFO, "main: MPI_Bcast: %s", error_str);

  if (tm_rank != 0) {
    memcpy(&fun, buff, sizeof(fun));
    memcpy(&precision, buff + sizeof(fun), sizeof(precision));
  }

  thread min_ub_receiver(tm_min_ub_receiver);
  min_ub_receiver.detach();

  thread min_ub_sender(tm_min_ub_sender);
  min_ub_sender.detach();

  this_thread::sleep_for(chrono::seconds(1));

  auto start = chrono::high_resolution_clock::now();

  box = tm_rank;
  while (box >= 0) {
    tm_find_box(tm_rank, tm_numprocs, fun.x, fun.y, box_x, box_y);
    minimize(fun.f, box_x, box_y, precision, tm_min_ub, tm_minimums);

    status = MPI_Send(&box, 1, MPI_INT, 0, TM_TAG_BOX, MPI_COMM_WORLD);
    MPI_Error_string(status, error_str, &error_len);
    syslog(LOG_INFO, "main: MPI_Send: %s", error_str);

    status =
        MPI_Recv(&box, 1, MPI_INT, 0, TM_TAG_BOX, MPI_COMM_WORLD, &status_mpi);
    MPI_Error_string(status, error_str, &error_len);
    syslog(LOG_INFO, "main: MPI_Recv: %s", error_str);
  }

  status = MPI_Reduce(&tm_min_ub, &min_ub, 1, MPI_DOUBLE, MPI_MIN, 0,
                      MPI_COMM_WORLD);
  MPI_Error_string(status, error_str, &error_len);
  syslog(LOG_INFO, "main: MPI_Reduce: %s", error_str);

  auto end = chrono::high_resolution_clock::now();

  if (tm_rank == 0) {
    // Displaying all potential minimizers
    copy(tm_minimums.begin(), tm_minimums.end(),
         ostream_iterator<minimizer>(cout, "\n"));
    cout << "Number of minimizers: " << tm_minimums.size() << endl;
    cout << "Upper bound for minimum: " << min_ub << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
  }

  MPI_Finalize();
  closelog();

  return 0;
}
