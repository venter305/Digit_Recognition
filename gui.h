extern const int APP_WIDTH;
extern const int APP_HEIGHT;
extern const int CANVAS_LEN;
extern uint8_t pixels[];

extern NeuralNetwork* nn;


void UpdateStats();
void Train(Button *btn);
void RandomizeImage(Button *btn);
void GuessUserInput(Button *btn);


void GUIInit(){
  glClearColor(0.8f,0.8f,0.8f,1.0f);

  //Panels
  std::shared_ptr<Panel> image = std::make_shared<Panel>(10,APP_HEIGHT-150,140,140);
  image->flipY();
  image->CreateTexture(28,28,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
  GraphicsEngine::guiMan.addElement(image,100);

  std::shared_ptr<Panel> canvas = std::make_shared<Panel>(10,APP_HEIGHT-400,CANVAS_LEN,CANVAS_LEN);
  for (int i=0;i<CANVAS_LEN*CANVAS_LEN*3;i++)
    pixels[i] = 0;
  canvas->CreateTexture(CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  GraphicsEngine::guiMan.addElement(canvas,400);

  std::shared_ptr<Panel> nnInCanvas = std::make_shared<Panel>(360,APP_HEIGHT-450,28*2,28*2);
  nnInCanvas->CreateTexture(28,28,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  nnInCanvas->flipY();
  GraphicsEngine::guiMan.addElement(nnInCanvas,401);

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

  std::shared_ptr<Panel> nnHidden = std::make_shared<Panel>(525,APP_HEIGHT-450,28*2,28*2);
  nnHidden->CreateTexture(width,height,GL_RGB,GL_UNSIGNED_BYTE,hidPixels);
  nnHidden->flipY();
  GraphicsEngine::guiMan.addElement(nnHidden,402);

  //Text
  const char* fontPath = "/usr/share/fonts/TTF/DejaVuSerif.ttf";
  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(160,APP_HEIGHT-50,20,"Label: ", fontPath),200);
  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(10,APP_HEIGHT-200,30,"Guess: ", fontPath),201);
  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(10,APP_HEIGHT-450,30,"Guess: ",fontPath),202);
  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(350,APP_HEIGHT-50,20,"Neural Network: ",fontPath),203);
  UpdateStats();

  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(300,APP_HEIGHT-425,15,"Input: ", fontPath),204);
  GraphicsEngine::guiMan.addElement(std::make_shared<Text>(450,APP_HEIGHT-425,15,"Hidden: ", fontPath),205);

  //Text Input Fields
  const char* numHid = std::to_string(nn->GetNumHidden()).c_str();

  float disabledColor[3] = {0.9f,0.9f,0.9f};
  float enabledColor[3] = {1,1,1};

  std::shared_ptr<TextInput> numInTxtInput = std::make_shared<TextInput>(675,APP_HEIGHT-167,65,25,numHid,[](TextInput *txtInput){
    std::string text = txtInput->GetText();
    int numHid = std::stoi(text);
    if (numHid != nn->GetNumHidden()){
      delete nn;
      nn = new NeuralNetwork(28*28,numHid,10);
    }
    UpdateStats();
  });
  numInTxtInput->setTextAlignment(TextInput::Left);
  numInTxtInput->SetTextMargins(5,-1);
  numInTxtInput->setTextSize(20);
  numInTxtInput->setStateColors(enabledColor,disabledColor);
  GraphicsEngine::guiMan.addElement(numInTxtInput,500);

  std::string lRateStr = std::to_string(nn->learningRate);
  lRateStr = lRateStr.substr(0,lRateStr.find(".")+3);

  std::shared_ptr<TextInput> learnRateTxtInput = std::make_shared<TextInput>(540,APP_HEIGHT-240,65,25,lRateStr,[](TextInput *txtInput){
    std::string text = txtInput->GetText();
    nn->learningRate = std::stof(text);
  });
  learnRateTxtInput->setTextAlignment(TextInput::Left);
  learnRateTxtInput->SetTextMargins(5,-1);
  learnRateTxtInput->setTextSize(20);
  learnRateTxtInput->setStateColors(enabledColor,disabledColor);
  GraphicsEngine::guiMan.addElement(learnRateTxtInput,501);

  std::string bSizeStr = std::to_string(nn->batchSize);

  std::shared_ptr<TextInput> bSizeTxtInput = std::make_shared<TextInput>(505,APP_HEIGHT-278,65,25,bSizeStr,[](TextInput *txtInput){
    std::string text = txtInput->GetText();
    nn->batchSize = std::stof(text);
  });
  bSizeTxtInput->setTextAlignment(TextInput::Left);
  bSizeTxtInput->SetTextMargins(5,-1);
  bSizeTxtInput->setTextSize(20);
  bSizeTxtInput->setStateColors(enabledColor,disabledColor);
  GraphicsEngine::guiMan.addElement(bSizeTxtInput,502);

  //Buttons
  std::shared_ptr<Button> randomize = std::make_shared<Button>(160,APP_HEIGHT-100,120,25,RandomizeImage);
  randomize->setText("Randomize Image");
  GraphicsEngine::guiMan.addElement(randomize,300);

  std::shared_ptr<Button> clear = std::make_shared<Button>(160,APP_HEIGHT-375,50,25,[](Button *btn){
    for (int i=0;i<CANVAS_LEN*CANVAS_LEN*3;i++)
      pixels[i] = 0;
    auto canvasPanel = std::static_pointer_cast<Panel>(GraphicsEngine::guiMan.elements[400]);
    canvasPanel->UpdateTexture(0,0,CANVAS_LEN,CANVAS_LEN,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  });
  clear->setText("Clear");
  GraphicsEngine::guiMan.addElement(clear,301);

  std::shared_ptr<Button> guessBtn = std::make_shared<Button>(160,APP_HEIGHT-315,50,25,GuessUserInput);
  guessBtn->setText("Guess");
  GraphicsEngine::guiMan.addElement(guessBtn,302);

  std::shared_ptr<Button> train = std::make_shared<Button>(350,APP_HEIGHT-350,50,25,Train);
  train->setText("Train");
  GraphicsEngine::guiMan.addElement(train,303);

  std::shared_ptr<Button> reset = std::make_shared<Button>(425,APP_HEIGHT-350,50,25,[](Button* btn){
    int numHidden = nn->GetNumHidden();
    delete nn;
    nn = new NeuralNetwork(28*28,numHidden,10);
    UpdateStats();
  });
  reset->setText("Reset");
  GraphicsEngine::guiMan.addElement(reset,304);

}
