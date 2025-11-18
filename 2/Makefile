CXX = mpic++
CXXFLAGS = -O3 -std=c++11

# Change 'main.cpp' to 'mpi.cpp'
SRCS = mpi.cpp load.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = main

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(EXEC)

test: all