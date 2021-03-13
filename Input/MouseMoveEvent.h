#pragma once

#include "Event.h"

class MouseMoveEvent : public Event {

public:

  MouseMoveEvent(double,double);
  ~MouseMoveEvent();

  double GetMouseX() {return mosPosX;};
  double GetMouseY() {return mosPosY;};

private:
  double mosPosX = 0;
  double mosPosY = 0;
};
