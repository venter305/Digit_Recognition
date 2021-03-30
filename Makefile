define IncludePaths
GraphicsEngine
GraphicsEngine/GUI
GraphicsEngine/Input
endef

_OBJ=$(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(foreach dir,$(IncludePaths),$(patsubst %.cpp,%.o,$(wildcard $(dir)/*.cpp)))

OBJ=$(patsubst %,./obj/%,$(_OBJ))

INCLUDE=-I./eigen -I/usr/include/freetype2

LFLAGS=-lGLEW -lglfw -lGL -lfreetype

CXXFLAGS=-O2 $(LFLAGS) $(INCLUDE)

nnet: $(OBJ)
	g++ $(OBJ) -o nnet $(CXXFLAGS)


init:
	mkdir -p obj obj/dep $(foreach dir,$(IncludePaths),obj/$(dir))  $(foreach dir,$(IncludePaths),obj/dep/$(dir))

clean:
	rm -f ./nnet ./obj/*.o ./obj/dep/*.d $(foreach dir,$(IncludePaths),obj/$(dir)/*.o) $(foreach dir,$(IncludePaths),obj/dep/$(dir)/*.d)

./obj/dep/%.d: %.cpp
	@rm -f $@
	@g++ -MM $< $(CXXFLAGS) > $@.tmp
	@sed 's,\($(@F:.d=.o)\)[ :]*,$(patsubst obj/dep/%.d,./obj/%.o,$@) :,g' < $@.tmp > $@
	@sed -i '$$a \\tg++ -c $< -o $(patsubst obj/dep/%.d,./obj/%.o,$@) $(CXXFLAGS)' $@
	@rm -f $@.tmp


-include $(patsubst %,./obj/dep/%,$(_OBJ:.o=.d))
