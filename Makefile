INCLUDES = -Itlx -isystem ips2ra/include -IDySECT
CFLAGS = -std=c++17 -Wall -Wextra -pedantic -Werror $(INCLUDES)
LDFLAGS = sorter.o tlx/build/tlx/libtlx.a -lpthread
BITS = -DRIBBON_BITS=$(RIBBON_BITS)

OPTFLAGS = -O3 -DNDEBUG -march=native
DEFFLAGS = -O2 -march=native
DBGFLAGS = -g #-fsanitize=address

all: ribbon

sorter.o: sorter.cpp sorter.hpp minimal_hasher.hpp
	$(CXX) $(OPTFLAGS) $(CFLAGS) -Wno-unused-function -c -o $@ $<

bench:	sorter.o ribbon.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(OPTFLAGS) $(BITS) $(CFLAGS) -o bench$(RIBBON_BITS) ribbon.cpp $(LDFLAGS)

ribbon:	sorter.o ribbon.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(DEFFLAGS) $(BITS) $(CFLAGS) -o ribbon$(RIBBON_BITS) ribbon.cpp $(LDFLAGS)

debug:	sorter.o ribbon.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(DBGFLAGS) $(BITS) $(CFLAGS) -o debug$(RIBBON_BITS) ribbon.cpp $(LDFLAGS)

tests:	tests.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) -O2 $(CFLAGS) -o tests tests.cpp -lgtest $(LDFLAGS)

parbench: sorter.o parbench.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(OPTFLAGS) $(BITS) $(CFLAGS) -o parbench$(RIBBON_BITS) parbench.cpp $(LDFLAGS)

parrun: sorter.o parbench.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(DEFFLAGS) $(BITS) $(CFLAGS) -o parrun$(RIBBON_BITS) parbench.cpp $(LDFLAGS)

pardbg: sorter.o parbench.cpp *.hpp rocksdb/*.h tlx Makefile
	$(CXX) $(DBGFLAGS) $(BITS) $(CFLAGS) -o pardbg$(RIBBON_BITS) parbench.cpp $(LDFLAGS)

tlx:
	mkdir -p tlx/build; \
	cd tlx/build; \
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g0 -GNinja ..; \
	ninja
