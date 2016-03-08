# Makefile example for the X8II070 project:
# "Minimization of a binary continuous function with and
#  interval-based branch-and-bound procedure"
#
# Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr
# Version 1.2, 2013-03-11
#
# ChangeLog:
# Added path to Boost headers
# Added variable BINROOT 
.PHONY: clean

BINROOT=/comptes/goualard-f/local/bin

COMMON_SOURCES = interval.cpp minimizer.cpp functions.cpp
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

CXXFLAGS = -std=gnu++0x -Wall -I/comptes/goualard-f/local/include

MPICXX = $(BINROOT)/mpic++


all: optimization-seq optimization-mpi

optimization-seq: optimization-seq.cpp $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

optimization-mpi: optimization-mpi.cpp $(COMMON_OBJECTS)
	$(MPICXX) $(CXXFLAGS) -fopenmp -o $@ $< $(COMMON_OBJECTS) -lm

$(COMMON_OBJECTS): %.o: %.cpp %.h


clean:
	-rm optimization-seq optimization-mpi $(COMMON_OBJECTS)