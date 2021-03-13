#include "MouseMoveEvent.h"

MouseMoveEvent::MouseMoveEvent(double mouseX,double mouseY) : mosPosX(mouseX),mosPosY(mouseY){
  type = EventType::MouseCursor;
}

MouseMoveEvent::~MouseMoveEvent(){

}
