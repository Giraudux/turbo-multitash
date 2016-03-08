/*
  Functions --

  Some examples of functions to optimize with their known minimizer.

  How to add new functions:
  1/ Add the signature to functions.h
  2/ Add the name and initial domains to the unordered_map "functions"
     at the beginning of functions.cpp
  3/ Add the code of the function to functions.cpp

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#ifndef __functions_h__
#define __functions_h__

#include <string>
#include <unordered_map>

// Forward declaration of the interval class
class interval;

// Signature type of a binary function to minimize
typedef interval (*itvfun)(const interval& x, const interval& y);

// Type to gather the information needed to start optimizing a
// function chosen by the user
struct opt_fun_t {
  itvfun f;   // Pointer to the function to minimize
  interval x; // Initial domain for the 1st variable
  interval y; // Initial domain for the 2nd variable
};

// Three hump camel --
//   Minimum in box [-5,5]x[-5,5]: f(0,0) = 0
interval three_hump_camel(const interval& x, const interval& y);

//  Goldstein-Price --
//  Minimum in box [-2,2]x[-2,2]: f(0,-1) = 3
interval goldstein_price(const interval& x, const interval& y);

// Beale's function --
// Minimum in box [-4.5, 4.5]x[-4.5, 4.5]: f(3,0.5) = 0
interval beale(const interval& x, const interval& y);

// Booth's function --
// Minimum in box [-10, 10]x[-10, 10]: f(1,3) = 0
interval booth(const interval& x, const interval& y);

#define FUNCTION_AND_NAME(n,domx,domy) {#n, opt_fun_t{n,domx,domy}}

// Database of all functions to optimize with the initial box
// in which a minimizer is sought.
// This database is used to launch the B&B algorithm on a
// a function whose name is given as a string by the user.
extern std::unordered_map<std::string, opt_fun_t> functions;

#endif // __functions_h__
