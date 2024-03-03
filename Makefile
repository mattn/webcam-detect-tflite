TENSORFLOW_ROOT=/home/mattn/dev/tensorflow/tensorflow

# The path to the location which you built tensorflow-lite with cmake.
TENSORFLOW_CBUILD=$(TENSORFLOW_ROOT)/lite/cbuild
CXXFLAGS ?= -g -I $(TENSORFLOW_ROOT) \
	-I $(TENSORFLOW_CBUILD)/flatbuffers/include \
	-I $(TENSORFLOW_CBUILD)/xnnpack/include \
	-I $(TENSORFLOW_CBUILD)/cpuinfo/include \
	-I $(TENSORFLOW_CBUILD)/pthreadpool-source/include \
	`pkg-config --cflags opencv4 freetype2`
LDFLAGS ?= -L $(TENSORFLOW_CBUILD) \
	-L $(TENSORFLOW_CBUILD)/_deps/flatbuffers-build \
	-L $(TENSORFLOW_CBUILD)/_deps/xnnpack-build \
	-L $(TENSORFLOW_CBUILD)/_deps/fft2d-build \
	-L $(TENSORFLOW_CBUILD)/_deps/cpuinfo-build \
	-L $(TENSORFLOW_CBUILD)/pthreadpool \
    -ltensorflow-lite -lXNNPACK -lpthreadpool -lflatbuffers -lfft2d_fftsg2d -lcpuinfo -lstdc++ -ltensorflowlite_c \
	`pkg-config --libs opencv4 freetype2` -lpthread -ldl -lm -lfreetype

.PHONY: all clean

all: webcam-detector

webcam-detector: main.o
	gcc -O3 -o webcam-detector main.o $(LDFLAGS)

main.o : main.cxx
	g++ -c --std=c++14 main.cxx -O3 $(CXXFLAGS)

clean:
	rm -f webcam-detector *.o
