CXXFLAGS ?= -g -I /home/mattn/go/src/github.com/tensorflow/tensorflow \
	-I /home/mattn/go/src/github.com/tensorflow/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	`pkg-config --cflags opencv4 freetype2`
LDFLAGS ?= -L /home/mattn/go/src/github.com/tensorflow/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib \
    -ltensorflow-lite -lXNNPACK -lpthreadpool -lfft2d_fftsg2d -lcpuinfo -lstdc++ -ltensorflowlite_c \
	`pkg-config --libs opencv4 freetype2` -lpthread -ldl -lm -lfreetype

.PHONY: all clean

all: webcam-detector

webcam-detector: main.o
	gcc -O3 -o webcam-detector main.o $(LDFLAGS)

main.o : main.cxx
	g++ -c --std=c++11 main.cxx -O3 $(CXXFLAGS)

clean:
	rm -f webcam-detector
