/*
  Interval --

  Interval data type with double precision bounds. Only implements
  operators for the evaluation of polynomials in interval arithmetic.

  See:
  Interval analysis. Ramon Moore. Prentice-Hall, 1966.
  
  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
 */

#ifndef __interval_h__
#define __interval_h__

#include <iosfwd>
#include <limits>

class interval {
 public:
  // Default constructor: [-inf, +inf]
  interval(); 
  // Double to interval constructor: [v, v]
  interval(double v);
  // Interval [l, r]
  interval(double l, double r);

  // Left bound
  const double left(void) const;
  // Right bound
  const double right(void) const;
  // Width (right-left) of the interval
  double width(void) const;
  // Midpoint of the interval
  double mid(void) const;

  // Is the interval empty?
  bool empty() const;

private:
  double& lb(void);
  double& rb(void);
  double bounds[2];
};

interval operator+(const interval& I1, const interval& I2);
interval operator-(const interval& I1, const interval& I2);
interval operator*(const interval& I1, const interval& I2);

interval pow(const interval& I, unsigned int n);

// Output
std::ostream& operator<<(std::ostream& os, const interval& I);

#endif // __interval_h__
