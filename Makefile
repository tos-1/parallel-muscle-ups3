FILES := Paint.c _Paint.c create_ghost.c 
SRCS := $(addprefix src/,$(FILES))

HEADERS := Paint.h create_ghost.h #-Isrc/mas.h -Isrc/masHalos.h
INCL := $(addprefix src/,$(HEADERS))

OBJS := $(SRCS:.c=.o)
SHARED := _Paint.so
SHARED := $(addprefix ext/,$(SHARED))
PYTHON := python

# python3.6-config --ldflags and --cflags 
PY_INCL  := -I/usr/include/python3.6m
MPI4PY_INCLUDE = -I${shell ${PYTHON} -c 'import mpi4py; print( mpi4py.get_include() )'}
SO := -shared
MPICC = mpicc
CFLAGS = -std=c99 -fPIC -Wall -O2 
LIBS = -lm -lpthread -ldl -lutil -lpython3.6m

%.o: %.c $(INCL)
	$(MPICC) $(CFLAGS) $(PY_INCL) $(MPI4PY_INCLUDE) -c $< -o $@

$(SHARED): $(OBJS)
	$(MPICC) $(OBJS) $(LIBS) $(SO) -o $(SHARED)

clean:
	rm -f $(OBJS) $(SHARED)
