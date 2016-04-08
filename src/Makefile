# Makefile example for the X8II070 project:
# "Minimization of a binary continuous function with and
#  interval-based branch-and-bound procedure"
#
# Authors: Frederic Goualard <Frederic.Goualard@univ-nantes.fr
#          Alexis Giraudet <Alexis.Giraudet@etu.univ-nantes.fr>
#          Dennis Bordet <Dennis.Bordet@etu.univ-nantes.fr>
#
# Version 1.3, 2016-04-08
#
# ChangeLog:
# Added path to Boost headers
# Added variable BINROOT
# Removed variable BINROOT and use PATH instead

.PHONY: clean

PATH := $(PATH):/comptes/goualard-f/local/bin

COMMON_SOURCES = interval.cpp minimizer.cpp functions.cpp
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

CXXFLAGS = -std=c++11 -Wall -I/comptes/goualard-f/local/include -fopenmp

MPICXX = mpic++


all: optimization-seq optimization-omp optimization-omp2 optimization-mpi optimization-mpi2

optimization-seq: optimization-seq.cpp $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

optimization-omp: optimization-omp.cpp $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

optimization-omp2: optimization-omp2.cpp $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

optimization-mpi: optimization-mpi.cpp $(COMMON_OBJECTS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

optimization-mpi2: optimization-mpi2.cpp $(COMMON_OBJECTS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJECTS) -lm

$(COMMON_OBJECTS): %.o: %.cpp %.h

clean:
	-rm -f optimization-seq optimization-omp optimization-omp2 optimization-mpi optimization-mpi2 $(COMMON_OBJECTS)
