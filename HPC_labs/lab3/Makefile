CC=gcc
NVCC=nvcc
MPICC=mpicc

CFLAGS=-O2
OMPFLAGS=-fopenmp
CUDAFLAGS=-arch=sm_35


LIBS=-lm
CUDALIBS=-lcudart
STDLIBS=-lstdc++
PTHREADSLIBS=-lpthread
OPENCLLIBS=-lOpenCL

PROGRAMS= cuda mpi openmp mpi+openmp

all: ${PROGRAMS}

cuda: cuda.cu
	${NVCC} ${CUDAFLAGS} $< -o $@
mpi: mpi.c
	${MPICC} ${CFLAGS} $< -o $@ ${LIBS}
openmp: openmp.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@ ${LIBS}
mpi+openmp: mpi+openmp.c
	${MPICC} ${CFLAGS} ${OMPFLAGS} $< -o $@ ${LIBS}

run:
	./cuda
	mpirun -np 8 -machinefile ./machinefile ./mpi
	export OMP_NUM_THREADS=8
	./openmp
	export OMP_NUM_THREADS=4
	mpirun -np 2 -machinefile ./machinefile ./mpi+openmp

clean:
	rm *.o ${PROGRAMS}
