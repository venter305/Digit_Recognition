#include <iostream>
#include <fstream>
#include <time.h>

#include "MNISTDataLoader.h"
#include "NeuralNetwork.h"

#include "GraphicsEngine/graphicsEngine.h"

NeuralNetwork *nn;
MNISTDataLoader::MNISTImages testImages;
MNISTDataLoader::MNISTLabels testLabels;

float *testInputBuffer;

const int APP_WIDTH = 550;
const int APP_HEIGHT = 500;

const int CANVAS_LEN = 28;

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
     imagePixels[y*testImages.numCols*3+x*3] = testImages.buffer[imageIndex+y*testImages.numCols+x];
     imagePixels[y*testImages.numCols*3+x*3+1] = testImages.buffer[imageIndex+y*testImages.numCols+x];
     imagePixels[y*testImages.numCols*3+x*3+2] = testImages.buffer[imageIndex+y*testImages.numCols+x];
   }
  imagePanel->UpdateTexture(0,0,28,28,GL_RGB,GL_UNSIGNED_BYTE,imagePixels);
}


void GUIInit(){
  glClearColor(0.7f,0.7f,0.7f,1.0f);
  std::shared_ptr<Panel> image = std::make_shared<Panel>(10,APP_HEIGHT-150,140,140);
  image->flipY();
  image->CreateTexture(28,28,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
  // glBindTexture(GL_TEXTURE_2D,image->tex);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  GraphicsEngine::guiMan.addElement(image,100);

  const char* fontPath = "/usr/share/fonts/TTF/DejaVuSerif.ttf";
  std::shared_ptr<Text> label = std::make_shared<Text>(160,APP_HEIGHT-50,20,"Label: ", fontPath);
  label->setTextColor(1,1,1);
  GraphicsEngine::guiMan.addElement(label,200);

  std::shared_ptr<Text> guess = std::make_shared<Text>(10,APP_HEIGHT-200,30,"Guess: ", fontPath);
  guess->setTextColor(1,1,1);
  GraphicsEngine::guiMan.addElement(guess,201);

  std::shared_ptr<Button> randomize = std::make_shared<Button>(300,APP_HEIGHT-100,120,25,RandomizeImage);
  randomize->setText("Randomize Image");
  GraphicsEngine::guiMan.addElement(randomize,300);

  std::shared_ptr<Panel> canvas = std::make_shared<Panel>(10,APP_HEIGHT-400,140,140);
  for (int i=0;i<CANVAS_LEN*CANVAS_LEN*3;i++)
    pixels[i] = 0;
  canvas->CreateTexture(CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  GraphicsEngine::guiMan.addElement(canvas,400);

  std::shared_ptr<Panel> nnInCanvas = std::make_shared<Panel>(160,APP_HEIGHT-400,CANVAS_LEN*2,CANVAS_LEN*2);
  nnInCanvas->CreateTexture(CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  nnInCanvas->flipY();
  GraphicsEngine::guiMan.addElement(nnInCanvas,401);

  std::shared_ptr<Button> clear = std::make_shared<Button>(300,APP_HEIGHT-375,50,25,[](Button *btn){
    for (int i=0;i<CANVAS_LEN*CANVAS_LEN*3;i++)
      pixels[i] = 0;
    auto canvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[400]);
    canvasPanel->UpdateTexture(0,0,CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  });
  clear->setText("Clear");
  GraphicsEngine::guiMan.addElement(clear,301);

  std::shared_ptr<Text> canvasGuess = std::make_shared<Text>(10,APP_HEIGHT-450,30,"Guess: ",fontPath);
  canvasGuess->setTextColor(1,1,1);
  GraphicsEngine::guiMan.addElement(canvasGuess,202);

  std::shared_ptr<Button> guessBtn = std::make_shared<Button>(300,APP_HEIGHT-300,50,25,[](Button *btn){
    float input[28*28];
    int canvasDiff = 28-CANVAS_LEN;
    for (int y=0;y<28;y++){
      if (y<canvasDiff/2 || y > 27-canvasDiff/2)
        for(int x=0;x<28;x++)
          input[y*28+x] = 0;
       else
        for(int x=0;x<28;x++){
          if (x < canvasDiff/2 || x > 27-canvasDiff/2)
            input[y*28+x] = 0;
          else
            input[y*28+x] = pixels[((CANVAS_LEN-1)-(y-canvasDiff/2))*CANVAS_LEN*3+(x-canvasDiff/2)*3]/255.0f;
        }
    }

    // for (int y=0;y<28;y++){
    //   for (int x=0;x<28;x++){
    //       std::cout << (input[y*28+x]?'#':'*');
    //   }
    //   std::cout << ' ' << y << std::endl;
    // }

    int leftOffset = 0;
    int rightOffset = 0;

    for (int x=0;x<28;x++){
      bool isVal = false;
      for (int y=0;y<28;y++)
        if (input[y*28+x] != 0){
          isVal = true;
          break;
        }
      if (!isVal)
        leftOffset++;
      else
        break;
    }

    for (int x=27;x>=0;x--){
      bool isVal = false;
      for (int y=0;y<28;y++)
        if (input[y*28+x] != 0){
          isVal = true;
          break;
        }
      if (!isVal)
        rightOffset++;
      else
        break;
    }

    int diff = (leftOffset-rightOffset)/2;
    int unsignedDiff = diff;
    if (diff < 0) unsignedDiff *= -1;

    float tmpIn[28*28];

    for (int i=0;i<unsignedDiff;i++){
      for (int x=0;x<28;x++)
        for (int y=0;y<28;y++){
          if (x == (diff>=0)?27:0)
            tmpIn[y*28+x] = 0;
          else
            tmpIn[y*28+x] = input[y*28+x+((diff<0)?-1:1)];
        }
      for (int i=0;i<28*28;i++)
         input[i] = tmpIn[i];
    }



    std::cout << leftOffset << ' ' << rightOffset << std::endl;

    auto nnCanvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[401]);
    uint8_t nnCanvasInput[CANVAS_LEN*CANVAS_LEN*3];

    for(int y=0;y<CANVAS_LEN;y++)
      for (int x=0;x<CANVAS_LEN;x++){
        nnCanvasInput[y*CANVAS_LEN*3+x*3+0] = input[y*28+x]*255;
        nnCanvasInput[y*CANVAS_LEN*3+x*3+1] = input[y*28+x]*255;
        nnCanvasInput[y*CANVAS_LEN*3+x*3+2] = input[y*28+x]*255;
      }

    nnCanvasPanel->UpdateTexture(0,0,CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,nnCanvasInput);


    int guess = GuessNumber(input);
    auto canvasGuess = std::static_pointer_cast<Text>(GraphicsEngine::guiMan.elements[202]);
    canvasGuess->setText("Guess: "+std::to_string(guess));
  });
  guessBtn->setText("Guess");
  GraphicsEngine::guiMan.addElement(guessBtn,302);

}

bool mouseDown = false;

void CanvasDraw(int mouseX, int mouseY){
  auto canvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[400]);
  if (canvasPanel->checkBoundingBox(mouseX,mouseY) && mouseDown){
    int leftEdge = canvasPanel->xPos;
    int rightEdge = canvasPanel->width+canvasPanel->xPos;
    int bottomEdge = canvasPanel->yPos;
    int topEdge = canvasPanel->height+canvasPanel->yPos;

    int mouseCanvasX = (int)(mouseX-leftEdge)/(canvasPanel->width/CANVAS_LEN);
    int mouseCanvasY = (int)(mouseY-bottomEdge)/(canvasPanel->height/CANVAS_LEN);

    //std::cout << mouseCanvasX << ' ' << mouseCanvasY << std::endl;
    //mouseCanvasX = 28;
    for (int k=-1;k<2;k++){
      for (int j=-1;j<2;j++){
        for (int i=0;i<3;i++){
          float fadeVal = 1;
          int pixelX = (mouseCanvasX+j)*3;
          if (pixelX >= CANVAS_LEN*3)
            pixelX = (CANVAS_LEN-1)*3;
          int pixelY = (mouseCanvasY+k)*CANVAS_LEN*3;
          int index = (pixelY+pixelX+i);

          if (pixels[index] < 255)
            fadeVal = 0.9;
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

				for (int i=300;i<=302;i++){
					std::static_pointer_cast<Button>(GraphicsEngine::guiMan.elements[i])->clickAction(mouseX,mouseY);
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
	}
}

void Renderloop(GLFWwindow *window,double dTime){
  glClear(GL_COLOR_BUFFER_BIT);

  GraphicsEngine::guiMan.drawElements();
}

int main(){

  srand(time(NULL));

  MNISTDataLoader::MNISTImages images;
  images.file = "../MNIST_Database/train-images-idx3-ubyte";
  MNISTDataLoader::MNISTLabels labels;
  labels.file = "../MNIST_Database/train-labels-idx1-ubyte";
  MNISTDataLoader::LoadData(images,labels);

  testImages.file = "../MNIST_Database/t10k-images-idx3-ubyte";
  testLabels.file = "../MNIST_Database/t10k-labels-idx1-ubyte";
  MNISTDataLoader::LoadData(testImages,testLabels);

  nn = new NeuralNetwork(images.numRows*images.numCols,30,10);

  float *inputBuffer = new float[images.numRows*images.numCols*images.numImages];
  float *targetBuffer = new float[labels.numLabels*10];
  for (int i=0;i<images.numImages*images.numRows*images.numCols;i++)
    inputBuffer[i] = images.buffer[i]/255.0;

  for (int i=0;i<labels.numLabels;i++){
    for (int j=0;j<10;j++)
      targetBuffer[i*10+j] = 0;

    targetBuffer[i*10+labels.buffer[i]] = 1;
  }

  testInputBuffer = new float[testImages.numRows*testImages.numCols*testImages.numImages];
  float *testTargetBuffer = new float[testLabels.numLabels*10];
  for (int i=0;i<testImages.numImages*testImages.numRows*testImages.numCols;i++)
    testInputBuffer[i] = testImages.buffer[i]/255.0;

  for (int i=0;i<testLabels.numLabels;i++){
    for (int j=0;j<10;j++)
      testTargetBuffer[i*10+j] = 0;

    testTargetBuffer[i*10+testLabels.buffer[i]] = 1;
  }

  //std::cout << "Before Training: " << nn->CalculateAccuracy(testImages.numImages,testInputBuffer,testTargetBuffer)*100 << "%" << std::endl;

  nn->Train(1,10,60000,inputBuffer,targetBuffer,3,true);

  //std::cout << "\nAfter Training: " << nn->CalculateAccuracy(testImages.numImages,testInputBuffer,testTargetBuffer)*100 << "%" << std::endl;


  GraphicsEngine::Init(APP_WIDTH,APP_HEIGHT,"Digit Recognition",Renderloop);

  GUIInit();


  RandomizeImage();


  GraphicsEngine::input.onEvent = OnEvent;
  GraphicsEngine::Run();

  delete[] inputBuffer,testInputBuffer;
  delete[] targetBuffer,testTargetBuffer;

  return 0;
}
