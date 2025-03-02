OBJS =  Graph.o Reductions.o Alg.o Mis.o ControlUnit.o Util.o
all: mis

CC 	= g++
FLAGS	= -std=c++11 -O2 -c

mis: $(OBJS) $(HEADER)
	$(CC) -g -O2 -o mis $(OBJS)

Util.o: src/Util.cpp
	$(CC) $(FLAGS) src/Util.cpp

Graph.o: src/Graph.cpp
	$(CC) $(FLAGS) src/Graph.cpp

ControlUnit.o: src/ControlUnit.cpp
	$(CC) $(FLAGS) src/ControlUnit.cpp

Reductions.o: src/Reductions.cpp
	$(CC) $(FLAGS) src/Reductions.cpp

Alg.o: src/Alg.cpp
	$(CC) $(FLAGS) src/Alg.cpp

Mis.o: src/Mis.cpp
	$(CC) $(FLAGS) src/Mis.cpp

clean:
	rm -f mis $(OBJS)

count:
	wc -l src/*.cpp src/*.hpp
