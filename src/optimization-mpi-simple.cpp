/*
  Branch and bound algorithm to find the minimum of continuous binary
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>

#include <mpi.h>

#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;

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

void read_fun_precision(opt_fun_t &fun, double &precision) {
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

int main(int argc, char *argv[]) {
  int gsize, rank;
  // numprocs ?

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &gsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;

  if (rank == 0) {
    read_fun_precision(fun, precision);
    
  }
  
  
  //********************************scatter****************************************
  /*
  int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    MPI_Comm comm)

sendbuf
    Address of send buffer (choice, significant only at root). 
sendcount
    Number of elements sent to each process (integer, significant only at root). 
sendtype
    Datatype of send buffer elements (handle, significant only at root). 
recvcount
    Number of elements in receive buffer (integer). 
recvtype
    Datatype of receive buffer elements (handle). 
root
    Rank of sending process (integer). 
comm
    Communicator (handle). 
*/
    
  //MPI_Scatter(/*donnees*/,/*sz/numprocs*/, MPI_DOUBLE, /*partieDuTablocalJecrois*/, /*sz/numprocs*/, MPI_DOUBLE,0/*=root*/, MPI_COMM_WORLD);
  
  
  auto start = chrono::high_resolution_clock::now();
  minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums); 
  auto end = chrono::high_resolution_clock::now();  
  
  //*********************************reduce min******************************************
//page 94 du cours de Mr Goualard

/*
int MPI_Reduce(void *sendbuf, void *recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
    
sendbuf
    Address of send buffer (choice). 
count
    Number of elements in send buffer (integer). 
datatype
    Data type of elements of send buffer (handle). 
op
    Reduce operation (handle). 
root
    Rank of root process (integer). 
comm
    Communicator (handle).     
    
*/

//MPI_Reduce(&local_min, &min, int count, MPI_Datatype datatype, MPI_MIN, 0, MPI_COMM_WORLD);


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

  return 0;
}
