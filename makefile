HIPCC=/opt/rocm/hip/bin/hipcc
CXX_OPENCV=`pkg-config --cflags opencv` `pkg-config --libs opencv`
CXX_OPENMP=-Xcompiler -fopenmp
CXX_FFTW=-I/usr/local/include -L/usr/local/lib -lfftw3f
CXXFLAGS=-std=c++11 ${CXX_OPENMP} ${CXX_OPENCV} ${CXX_FFTW}
HIPFLAGS=-arch=sm_86 --disable-warnings
DST=convolution2D.out
SRC=$(wildcard *.cpp *.h)

all: clean ${DST}

${DST}: ${SRC}
	${HIPCC} ${CXXFLAGS} ${HIPFLAGS} -o $@ ${SRC}

.PHONY: clean
clean:
	rm -f *.out *.o
	echo Clean done
