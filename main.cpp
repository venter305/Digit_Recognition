#include <iostream>
#include <fstream>
#include <time.h>

#include "GraphicsEngine/graphicsEngine.h"

#include "MNISTData.h"
#include "NeuralNetwork.h"
#include "gui.h"


NeuralNetwork *nn;

MNISTImages testImages;
MNISTLabels testLabels;

float *inputBuffer;
float *targetBuffer;

float *testInputBuffer;
float *testTargetBuffer;

const int APP_WIDTH = 750;
const int APP_HEIGHT = 500;

const int CANVAS_LEN = 140;

uint8_t pixels[CANVAS_LEN*CANVAS_LEN*3];

int GuessNumber(float* imageBuffer, int offset = 0){
  float outputs[10];

  nn->CalculateOutputs(&imageBuffer[testImages.numRows*testImages.numCols*offset],outputs);

  float max = 0;
  int maxIndex = 0;
  for (int j=0;j<10;j++){
    if (outputs[j] >= max){
      max = outputs[j];
      maxIndex = j;
    }
  }

  return maxIndex;
}

void RandomizeImage(Button *btn = nullptr){
  int offset = rand()%testImages.numImages;
  int imageIndex = offset*testImages.numRows*testImages.numCols;
  int guessNum = GuessNumber(testInputBuffer,offset);
  auto guessText = std::static_pointer_cast<Text>(GraphicsEngine::guiMan.elements[201]);
  guessText->setText("Guess: "+std::to_string(guessNum));
  auto labelText = std::static_pointer_cast<Text>(GraphicsEngine::guiMan.elements[200]);
  labelText->setText("Label:"+std::to_string(testLabels.buffer[offset]));

  auto imagePanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[100]);
  uint8_t imagePixels[testImages.numRows*testImages.numCols*3];
  for (int y=0;y<testImages.numRows;y++)
   for (int x=0;x<testImages.numCols;x++){
     imagePixels[y*testImages.numCols*3+x*3] = testInputBuffer[imageIndex+y*testImages.numCols+x]*255;
     imagePixels[y*testImages.numCols*3+x*3+1] = testInputBuffer[imageIndex+y*testImages.numCols+x]*255;
     imagePixels[y*testImages.numCols*3+x*3+2] = testInputBuffer[imageIndex+y*testImages.numCols+x]*255;
   }
  imagePanel->UpdateTexture(0,0,28,28,GL_RGB,GL_UNSIGNED_BYTE,imagePixels);
}

void UpdateStats(){
  std::string statsString = "NeuralNetwork:\n"
    "\tNumber of Inputs: " + std::to_string(nn->GetNumInputs()) + "\n"
    "\tNumber of Outputs: " + std::to_string(nn->GetNumOutputs()) + "\n"
    "\tNumber of Hidden Neurons: \n"
    "\n"
    "\tLearning Rate: \n"
    "\tBatch Size: \n"
    "\tAccuracy: " + std::to_string(nn->CalculateAccuracy(testImages.numImages,testInputBuffer,testTargetBuffer)*100) + "%";
  auto statsText = std::static_pointer_cast<Text>(GraphicsEngine::guiMan.elements[203]);
  statsText->setText(statsString);

  int width = std::ceil(std::sqrt(nn->GetNumHidden()));
  int height = std::round(std::sqrt(nn->GetNumHidden()));
  uint8_t hidPixels[width*height*3];
  double *hidden = nn->GetHidden();
  for (int i=0;i<width*height;i++){
    int value = hidden[i]*255;
    hidPixels[i*3+0] = value;
    hidPixels[i*3+1] = value;
    hidPixels[i*3+2] = value;
  }

  auto nnHidPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[402]);
  nnHidPanel->CreateTexture(width,height,GL_RGB,GL_UNSIGNED_BYTE,hidPixels);
}

void Train(Button *btn){
  nn->Train(1,nn->batchSize,60000,inputBuffer,targetBuffer,nn->learningRate,nn->lambdaVal,true);

  UpdateStats();
}

void GuessUserInput(Button *btn){
  float input[28*28];
  int canvasDiff = 28-CANVAS_LEN;
  //Downscale input
  for (int y=0;y<28;y++){
    for (int x=0;x<28;x++){
      if (y<canvasDiff/2 || y > 27-canvasDiff/2)
        input[y*28+x] = 0;
     else
        if (x < canvasDiff/2 || x > 27-canvasDiff/2)
          input[y*28+x] = 0;
        else{
          float average = 0;
          int size = (CANVAS_LEN/28);
          for (int subY=0;subY<size;subY++)
            for (int subX=0;subX<size;subX++){
              int sY = ((28-y)*size+(subY-size))*CANVAS_LEN*3;
              int sX = (x*size+subX)*3;

              average += pixels[sY+sX];
            }
          input[y*28+x] = (average/(size*size))/255.0f;
        }
      }
  }

  CenterImage(input);

  auto nnCanvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[401]);
  uint8_t nnCanvasInput[28*28*3];

  for(int y=0;y<28;y++)
    for (int x=0;x<28;x++){
      nnCanvasInput[y*28*3+x*3+0] = input[y*28+x]*255;
      nnCanvasInput[y*28*3+x*3+1] = input[y*28+x]*255;
      nnCanvasInput[y*28*3+x*3+2] = input[y*28+x]*255;
    }

  nnCanvasPanel->UpdateTexture(0,0,28,28,GL_RGB,GL_UNSIGNED_BYTE,nnCanvasInput);

  int guess = GuessNumber(input);
  auto canvasGuess = std::static_pointer_cast<Text>(GraphicsEngine::guiMan.elements[202]);
  canvasGuess->setText("Guess: "+std::to_string(guess));

  auto nnHidPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[402]);
  int width = std::ceil(std::sqrt(nn->GetNumHidden()));
  int height = std::round(std::sqrt(nn->GetNumHidden()));
  uint8_t hidPixels[width*height*3];
  double *hidden = nn->GetHidden();
  for (int i=0;i<width*height;i++){
    int value = hidden[i]*255;
    hidPixels[i*3+0] = value;
    hidPixels[i*3+1] = value;
    hidPixels[i*3+2] = value;
  }
  nnHidPanel->UpdateTexture(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,hidPixels);
}


bool mouseDown = false;

void CanvasDraw(int mouseX, int mouseY){
  auto canvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[400]);
  if (canvasPanel->checkBoundingBox(mouseX,mouseY) && mouseDown){
    int leftEdge = canvasPanel->xPos;
    //int rightEdge = canvasPanel->width+canvasPanel->xPos;
    int bottomEdge = canvasPanel->yPos;
    //int topEdge = canvasPanel->height+canvasPanel->yPos;

    int mouseCanvasX = (int)(mouseX-leftEdge)/(canvasPanel->width/CANVAS_LEN);
    int mouseCanvasY = (int)(mouseY-bottomEdge)/(canvasPanel->height/CANVAS_LEN);

    int lineWidth = 15;

    for (int k=lineWidth/-2;k<=lineWidth/2;k++){
      double xWidth = std::sqrt(pow(0.5,2)-pow((double)k/lineWidth,2))*(lineWidth);
      xWidth = round(xWidth);
      for (int j=xWidth*-1;j<xWidth;j++){
        for (int i=0;i<3;i++){
          float fadeVal = 1;
          int pixelX = (mouseCanvasX+j)*3;
          if (pixelX >= CANVAS_LEN*3)
            pixelX = (CANVAS_LEN-1)*3;
          int pixelY = (mouseCanvasY+k)*CANVAS_LEN*3;
          int index = (pixelY+pixelX+i);

          if (pixels[index] < 255)
            fadeVal = ((j>4||j<-4)||(k>4||k<-4)?1:1);
          if (pixelX < CANVAS_LEN*CANVAS_LEN*3 && pixelX >= 0 && pixelY < CANVAS_LEN*CANVAS_LEN*3 && pixelY >= 0)
            pixels[index] = 255*((j||k)?fadeVal:1);
        }
      }
    }
    canvasPanel->UpdateTexture(0,0,CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  }
}

void OnEvent(Event &ev){
  switch (ev.GetType()){
    case Event::MouseButton:
		{
			MouseButtonEvent::ButtonType btnType = static_cast<MouseButtonEvent*>(&ev)->GetButtonType();
			MouseButtonEvent::ButtonState btnState = static_cast<MouseButtonEvent*>(&ev)->GetButtonState();
		  double mouseX = static_cast<MouseButtonEvent*>(&ev)->GetMouseX();
			double mouseY = static_cast<MouseButtonEvent*>(&ev)->GetMouseY();
			if(btnType == MouseButtonEvent::ButtonType::Left && btnState == MouseButtonEvent::ButtonState::Pressed){
        for (auto element : GraphicsEngine::guiMan.elements){
					element.second->clickAction(mouseX,mouseY);
				}
        mouseDown = true;
        CanvasDraw(mouseX,mouseY);
			}
      if(btnType == MouseButtonEvent::ButtonType::Left && btnState == MouseButtonEvent::ButtonState::Released){
        mouseDown = false;
      }
			break;
		}
    case Event::MouseCursor:
      {
        double mouseX = static_cast<MouseMoveEvent*>(&ev)->GetMouseX();
        double mouseY = static_cast<MouseMoveEvent*>(&ev)->GetMouseY();
        mouseY = APP_HEIGHT-mouseY;

        CanvasDraw(mouseX,mouseY);

        break;
      }
    case Event::Character:
      {
        unsigned int charCode = static_cast<CharEvent*>(&ev)->GetCharCode();
        for (int i=500;i<=502;i++){
          auto txtInput = std::static_pointer_cast<TextInput>(GraphicsEngine::guiMan.elements[i]);
          if (txtInput->enabled)
            txtInput->addCharacter((char)charCode);
        }
        break;
      }
    case Event::Key:
    {
      int keyCode = static_cast<KeyEvent*>(&ev)->GetKeyCode();
      KeyEvent::KeyState keyState = static_cast<KeyEvent*>(&ev)->GetKeyState();

      for (int i=500;i<=502;i++){
        auto txtInput = std::static_pointer_cast<TextInput>(GraphicsEngine::guiMan.elements[i]);
        if (txtInput->enabled){
          if (keyCode == GLFW_KEY_BACKSPACE && keyState == KeyEvent::KeyState::Pressed)
            txtInput->removeCharacter();
          if (keyCode == GLFW_KEY_ENTER && keyState == KeyEvent::KeyState::Pressed){
            txtInput->setEnabled(false);
          }
        }
      }
      break;
    }
    default:
      break;
	}
}

void Renderloop(GLFWwindow *window,double dTime){
  glClear(GL_COLOR_BUFFER_BIT);

  GraphicsEngine::guiMan.drawElements();
}

int main(){

  srand(time(NULL));

  MNISTImages images;
  MNISTLabels labels;

  images.file = "../MNIST_Database/train-images-idx3-ubyte";
  labels.file = "../MNIST_Database/train-labels-idx1-ubyte";
  LoadMNISTData(images,labels);

  testImages.file = "../MNIST_Database/t10k-images-idx3-ubyte";
  testLabels.file = "../MNIST_Database/t10k-labels-idx1-ubyte";
  LoadMNISTData(testImages,testLabels);

  nn = new NeuralNetwork(images.numRows*images.numCols,30,10);

  inputBuffer = new float[images.numRows*images.numCols*images.numImages];
  targetBuffer = new float[labels.numLabels*10];
  ProcessMNISTData(images,labels,inputBuffer,targetBuffer);

  testInputBuffer = new float[testImages.numRows*testImages.numCols*testImages.numImages];
  testTargetBuffer = new float[testLabels.numLabels*10];
  ProcessMNISTData(testImages,testLabels,testInputBuffer,testTargetBuffer);

  GraphicsEngine::Init(APP_WIDTH,APP_HEIGHT,"Digit Recognition",Renderloop);

  GUIInit();

  RandomizeImage();

  GraphicsEngine::input.onEvent = OnEvent;
  GraphicsEngine::Run();

  delete[] inputBuffer;
  delete[] testInputBuffer;
  delete[] targetBuffer;
  delete[] testTargetBuffer;

  return 0;
}
