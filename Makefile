CX = g++
CXFLAGS = -g -Wall 

CVFLAGS = `pkg-config opencv4 --cflags --libs`

BUILDFLAGS = $(CVFLAGS)

TARGET = objectdetect
OBJS = main.o yolo.o
$(TARGET) :  $(OBJS)
	$(CX) $(CXFLAGS) -o $(TARGET) $(OBJS) $(BUILDFLAGS) 
main.o : main.cpp
	$(CX) $(CXFLAGS) -c main.cpp $(BUILDFLAGS) 
yolo.o : yolo.hpp yolo.cpp
	$(CX) $(CXFLAGS) -c yolo.cpp $(CVFLAGS)

.PHONY: all clean
all: $(TARGET)

clean:
	rm -rf $(TARGET) $(OBJS)


