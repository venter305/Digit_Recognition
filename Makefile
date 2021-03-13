define IncludePaths
GraphicsEngine
GraphicsEngine/GUI
GraphicsEngine/Input
endef

_OBJ=$(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(foreach dir,$(IncludePaths),$(patsubst %.cpp,%.o,$(wildcard $(dir)/*.cpp)))

OBJ=$(patsubst %,./obj/%,$(_OBJ))

INCLUDE=-I./eigen -I/usr/include/freetype2

LFLAGS=-lGLEW -lglfw -lGL -lfreetype

CFLAGS=-O2

nnet: MNISTDataLoader.h $(OBJ)
	g++ $(OBJ) $(INCLUDE) $(LFLAGS) -o nnet $(CFLAGS)

obj/%.o: %.cpp %.h
	g++ -c $< -o $@ $(INCLUDE) $(CFLAGS)

obj/%.o: %.cpp
	g++ -c  -o $@ $< $(INCLUDE) $(CFLAGS)

init:
	mkdir -p obj $(foreach dir,$(IncludePaths),obj/$(dir))

clean:
	rm -f ./nnet ./obj/*.o $(foreach dir,$(IncludePaths),obj/$(dir)/*.o)
