/* 
 * Some examples of functions to optimize with their known minimizer.

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15

*/

#include "interval.h"
#include "functions.h"

// Database of all functions to optimize with the initial box
// in which a minimizer is sought.
std::unordered_map<std::string, opt_fun_t> functions {
  FUNCTION_AND_NAME(three_hump_camel,interval(-5,5),interval(-5,5)),
    FUNCTION_AND_NAME(goldstein_price,interval(-2,2),interval(-2,2)),
    FUNCTION_AND_NAME(beale,interval(-4.5,4.5),interval(-4.5,4.5)),
    FUNCTION_AND_NAME(booth,interval(-10,10),interval(-10,10))
};

// Three hump camel --
//   Minimum in box [-5,5]x[-5,5]: f(0,0) = 0
interval three_hump_camel(const interval& x, const interval& y)
{ // Function scaled by factor 600 to avoid fractional coefficients
  return 1200*pow(x,2)-630*pow(x,4)+100*pow(x,6)+x*y+pow(y,2);
}

//  Goldstein-Price --
//  Minimum in box [-2,2]x[-2,2]: f(0,-1) = 3
interval goldstein_price(const interval& x, const interval& y)
{
  return (1 + pow(x+y+1,2)*(19-14*x+3*pow(x,2) - 
			    14*y+6*x*y+3*pow(y,2)))*
    (30+pow(2*x-3*y,2)*(18-32*x+12*pow(x,2)+48*y
			-36*x*y+27*pow(y,2)));
}

// Beale's function --
// Minimum in box [-4.5, 4.5]x[-4.5, 4.5]: f(3,0.5) = 0
interval beale(const interval& x, const interval& y)
{
  return pow(1.5-x+x*y,2)+pow(2.25-x+x*pow(y,2),2)+pow(2.625-x+x*pow(y,3),2);
}

// Booth's function --
// Minimum in box [-10, 10]x[-10, 10]: f(1,3) = 0
interval booth(const interval& x, const interval& y)
{
  return pow(x+2*y-7,2)+pow(2*x+y-5,2);
}

