
LRL1_DIR = $(APPS)/lrl1
LRL1_SRC = $(wildcard $(LRL1_DIR)/*.cpp)
LRL1_HDR = $(wildcard $(LRL1_DIR)/*.hpp)
LRL1_OBJ = $(LRL1_SRC:%.cpp=%.o)
LRL1 = $(BIN)/lrl1_main

lrl1: $(LRL1)

$(LRL1): $(LRL1_OBJ) $(PS_CLIENT)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(LRL1_OBJ): %.o: %.cpp $(LRL1_HDR)
	$(CXX) $(CXXFLAGS) -I$(LRL1_DIR) $(INCFLAGS) -c $< -o $@

lrl1_clean:
	rm -f $(LRL1_OBJ) $(LRL1)

.PHONY: lrl1 svm_sdca_clean
