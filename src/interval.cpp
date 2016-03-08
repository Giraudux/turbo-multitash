/*
  Interval --

  Interval data type with double precision bounds. Only implements
  operators for the evaluation of polynomials in interval arithmetic.

  See:
  Interval analysis. Ramon Moore. Prentice-Hall, 1966.
  
  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include "interval.h"
#include <iostream>
#include <fenv.h>
#include <cmath>
#include <limits>

using namespace std;

const double inf = numeric_limits<double>::infinity();
const double NaN = numeric_limits<double>::quiet_NaN();
const double nsmall = (1.0-2.0*numeric_limits<double>::epsilon());
const double psmall = (1.0+2.0*numeric_limits<double>::epsilon());


const double interval::left(void) const
{
  return bounds[0];
}

const double interval::right(void) const
{
  return bounds[1];
}

double& interval::lb(void)
{
  return bounds[0];
}

double& interval::rb(void)
{
  return bounds[1];
}

bool interval::empty() const
{
  return left() > right();
}

inline double min(double a, double b)
{
  if (::isnan(a) || ::isnan(b)) {
    return NaN;
  }
  return (a <= b) ? a : b;
}

inline double max(double a, double b)
{
  if (::isnan(a) || ::isnan(b)) {
    return NaN;
  }
  return (a >= b) ? a : b;
}


inline void round_downward(void)
{
  fesetround(FE_DOWNWARD);
}

inline void round_upward(void)
{
  fesetround(FE_UPWARD);
}

inline void round_nearest(void)
{
  fesetround(FE_TONEAREST);
}

// Mimicking downward rounding for functions that do not obey 
// the current rounding direction of the FPU
inline double round_down(double d)
{
  if (d == inf) {
    return numeric_limits<double>::max();
  } else {
    return ((d<0.0) 
	    ? psmall*d-numeric_limits<double>::min() 
            : nsmall*d-numeric_limits<double>::min());
  }
}

// Mimicking upward rounding for functions that do not obey 
// the current rounding direction of the FPU
inline double round_up(double d)
{
  if (d == -inf) {
    return -numeric_limits<double>::max();
  } else {
    return ((d<0.0) 
	    ? nsmall*d+numeric_limits<double>::min() 
            : psmall*d+numeric_limits<double>::min());
  }
}

interval::interval() 
{
  lb() = -inf;
  rb() = inf;
}

interval::interval(double v) 
{
  lb() = v;
  rb() = v;
}

interval::interval(double l, double r)
{
  lb() = l;
  rb() = r;
}

interval operator+(const interval& I1, const interval& I2)
{
  round_downward();
  double l = I1.left() + I2.left();
  round_upward();
  double r = I1.right() + I2.right();
  round_nearest();
  return interval(l,r);
}

interval operator-(const interval& I1, const interval& I2)
{
  round_downward();
  double l = I1.left() - I2.right();
  round_upward();
  double r = I1.right() - I2.left();
  round_nearest();
  return interval(l,r);
}

interval operator*(const interval& I1, const interval& I2)
{
  round_downward();
  double l = min(min(I1.left()*I2.left(),I1.left()*I2.right()),
		 min(I1.right()*I2.left(),I1.right()*I2.right()));
  round_upward();
  double r = max(max(I1.left()*I2.left(),I1.left()*I2.right()),
		 max(I1.right()*I2.left(),I1.right()*I2.right()));
  round_nearest();
  return interval(l,r);
}

interval pow(const interval& I, unsigned int n)
{
  if (n == 0) {
    return interval(1.0);
  }
  if (n == 1) {
    return I;
  }

  if ((n % 2) == 0) { // n is even?
    if (I.left() >= 0) { // I is positive?
      double l = round_down(std::pow(I.left(),n));
      double r = round_up(std::pow(I.right(),n));
      return interval(l,r);
    } else {
      if (I.right() <= 0) { // I is negative?
	double l = round_down(std::pow(I.right(),n));
	double r = round_up(std::pow(I.left(),n));
	return interval(l,r);
      } else { // I straddles 0
	double rl = round_up(std::pow(I.right(),n));
	double rr = round_up(std::pow(I.left(),n));
	return interval(0,max(rl,rr));
      }
    }
  }
  // n is odd
  double l = round_down(std::pow(I.left(),n));
  double r = round_up(std::pow(I.right(),n));
  return interval(l,r);
}

double interval::width() const
{
  if (empty()) {
    return NaN;
  }
  if (::isinf(left()) || ::isinf(right())) {
    return inf;
  }
  return right()-left();
}

double interval::mid() const
{
  if (empty()) {
    return NaN;
  }
  if (left() == -inf) {
    return -numeric_limits<double>::max();
  }
  if (right() == inf) {
    return numeric_limits<double>::max();
  }
  double middle = 0.5*(left()+right());
  
  if (::isinf(middle)) {
    return (0.5*left() + 0.5*right()); 
  } else {
    return middle;
  }
}

ostream& operator<<(ostream& os, const interval& I)
{
  if (I.empty()) {
    os << "[Empty]";
  } else {
    round_downward();
    os << "[" << I.left();
    round_upward();
    os << ", " << I.right() << "]";
  }
  return os;
}
