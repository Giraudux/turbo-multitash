/*
  Minimizer --

  A minimizer is a small box for which the function potentially has
  a smaller value than the current upper bound.

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <iostream>
#include <boost/format.hpp>
#include "interval.h"
#include "minimizer.h"

using namespace std;

ostream& operator<<(ostream& os, const minimizer& m)
{
  os << boost::format("{(%1%,%2%), [%3%, %4%]}") 
    % m.xmin % m.ymin % m.lbmin % m.ubmin;
  return os;
}
