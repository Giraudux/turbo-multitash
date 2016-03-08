/*
  Minimizer --

  A minimizer is a small box for which the function potentially has
  a smaller value than the current upper bound.

  The minimizer contains the two intervals xmin and ymin defining the box, the
  lower bound lbmin of the minimum (f(xmin,ymin) is certainly greater 
  or equal to lbmin), and the upper bound ubmin of the minimum (f(xmin,ymin)
  is certainly smaller or equal to ubmin).

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#ifndef __minimizer_h__
#define __minimizer_h__

#include <iosfwd>
#include <set>

class interval;

struct minimizer {
  interval xmin;
  interval ymin;
  double lbmin;
  double ubmin;
};

struct less_minimizer {
  bool operator()(const minimizer& m1, const minimizer& m2) const
  {
    return m1.lbmin <= m2.lbmin;
  }
};

std::ostream& operator<<(std::ostream& os, const minimizer& m);

// Set of minimizers sorted in increasing value of the lower bound of
// the minimum.
typedef std::multiset<minimizer,less_minimizer> minimizer_list;

#endif // __minimizer_h__
