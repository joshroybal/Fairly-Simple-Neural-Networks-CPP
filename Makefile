CXXFLAGS = -std=c++98 -pedantic-errors -Wall -Wextra -Werror -O2
LFLAGS = -static -s
SRC = $(wildcard *.cpp)
HDR = $(wildcard *.hpp)
OBJ = $(SRC:.cpp=.o)
BIN = driver

$(BIN): $(OBJ)
	$(CXX) -o $@ $^ $(LFLAGS)

%.o: %.cpp $(HDR)
	$(CXX) -c $< $(CXXFLAGS)

.PHONY : clean
clean:
	$(RM) $(BIN) $(OBJ) *~
