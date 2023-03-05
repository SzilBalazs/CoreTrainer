CXX = g++
TARGET_FLAGS =
OBJECT_DIR = objects
SOURCES := $(wildcard *.cpp)
OBJECTS := $(patsubst %.cpp, $(OBJECT_DIR)/%.o, $(SOURCES))

CXXFLAGS = -g -std=c++20 -O3 -ffast-math -funroll-loops -flto -pthread -Wall -Wno-unused-result -march=native -fsanitize=undefined
EXE = CoreTrainer

default: $(EXE)

$(OBJECTS): | $(OBJECT_DIR)

$(OBJECT_DIR):
	@mkdir $(OBJECT_DIR)

$(OBJECT_DIR)/%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

$(EXE): $(OBJECTS)
	@echo Linking $(EXE)
	@$(CXX) $(TARGET_FLAGS) $(CXXFLAGS) -o $@ $^
	@echo Build has finished.

all: clean build

build: $(EXE)

clean:
	@rm -r objects $(EXE) || true

.PHONY: all build clean