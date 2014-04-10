
LRL2_DIR = $(APPS)/lrl2
LRL2_SRC = $(wildcard $(LRL2_DIR)/*.cpp)
LRL2_HDR = $(wildcard $(LRL2_DIR)/*.hpp)
LRL2_OBJ = $(LRL2_SRC:%.cpp=%.o)
LRL2 = $(BIN)/lrl2_main

lrl2: $(LRL2)

$(LRL2): $(LRL2_OBJ) $(PS_CLIENT)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(LRL2_OBJ): %.o: %.cpp $(LRL2_HDR)
	$(CXX) $(CXXFLAGS) -I$(LRL2_DIR) $(INCFLAGS) -c $< -o $@

lrl2_clean:
	rm -f $(LRL2_OBJ) $(LRL2)

.PHONY: lrl2 svm_sdca_clean
