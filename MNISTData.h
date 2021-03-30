#pragma once
#include <fstream>

struct MNISTImages{
  const char *file;
  uint8_t *buffer;
  int numImages;
  int numRows;
  int numCols;

  ~MNISTImages(){
    delete[] buffer;
  }
};

struct MNISTLabels{
  const char *file;
  uint8_t *buffer;
  int numLabels;

  ~MNISTLabels(){
    delete[] buffer;
  }
};

void CenterImage(float *input){
  int leftOffset = 0;
  int rightOffset = 0;
  int topOffset = 0;
  int bottomOffset = 0;

  bool isValL = false;
  bool isValR = false;
  bool isValT = false;
  bool isValB = false;
  for (int i=0;i<28;i++){
    for (int k=0;k<28;k++){
      if (input[k*28+i] != 0){
        isValL = true;
      }
      if (input[k*28+(27-i)] != 0)
        isValR = true;

      if (input[i*28+k] != 0){
        isValT = true;
      }
      if (input[(27-i)*28+k] != 0){
        isValB = true;
      }
    }
    if (!isValL) leftOffset++;
    if (!isValR) rightOffset++;
    if (!isValT) topOffset++;
    if (!isValB) bottomOffset++;
    if(isValL && isValR && isValT && isValB) break;
  }

  int xDiff = (leftOffset-rightOffset)/2;
  int yDiff = (topOffset-bottomOffset)/2;
  float tmpIn[28*28];
  for (int y=0;y<28;y++){
    for (int x=0;x<28;x++){
      int index = (y+yDiff)*28+x+(xDiff);
      if (index < 0 || index > 28*28)
        tmpIn[y*28+x] = 0;
      else
        tmpIn[y*28+x] = input[index];
    }
  }
  for (int i=0;i<28*28;i++)
     input[i] = tmpIn[i];

}

void ProcessMNISTData(MNISTImages &images, MNISTLabels &labels, float* inputBuf, float* labelBuf){
  for (int i=0;i<images.numImages;i++){
    float tmpBuf[28*28];
    for (int j=0;j<28*28;j++)
      tmpBuf[j] = images.buffer[i*28*28+j]/255.0f;
    CenterImage(tmpBuf);
    for (int j=0;j<28*28;j++)
      inputBuf[i*28*28+j] = tmpBuf[j];
  }

  for (int i=0;i<labels.numLabels;i++){
    for (int j=0;j<10;j++)
      labelBuf[i*10+j] = 0;

    labelBuf[i*10+labels.buffer[i]] = 1;
  }
}

void LoadMNISTData(MNISTImages &images, MNISTLabels &labels){

  std::ifstream trainImages,trainLabels;

  trainImages.open(images.file,std::ios::binary);
  trainLabels.open(labels.file,std::ios::binary);

  if(!trainImages || !trainLabels){
    //std::cout << "No Image Files" << std::endl;
    return;
  }

  //std::cout << "Files Loaded" << std::endl;

  char data[4];
  trainImages.read(data,4);

  // std::cout << "\nImages:" << std::endl;
  // std::cout << "\tMagic Number: " << (int)data[0] << ' ' << (int)data[1] << ' ' << (int)data[2] << ' ' << (int)data[3] << std::endl;

  trainImages.read(data,4);
  images.numImages = (unsigned char)data[0]*0x1000000+(unsigned char)data[1]*10000+
                  (unsigned char)data[2]*0x100+(unsigned char)data[3];

  //std::cout << "\tNumber of Images: " << images.numImages << std::endl;

  trainImages.read(data,4);
  images.numRows = (unsigned char)data[0]*0x1000000+(unsigned char)data[1]*10000+
                (unsigned char)data[2]*0x100+(unsigned char)data[3];

  //std::cout << "\tNumber of Rows: " << images.numRows << std::endl;

  trainImages.read(data,4);
  images.numCols = (unsigned char)data[0]*0x1000000+(unsigned char)data[1]*10000+
                (unsigned char)data[2]*0x100+(unsigned char)data[3];

  // std::cout << "\tNumber of Columns: " << images.numCols << std::endl;
  // std::cout << std::endl;

  trainLabels.read(data,4);

  // std::cout << "\nLabels:" << std::endl;
  // std::cout << "\tMagic Number: " << (int)data[0] << ' ' << (int)data[1] << ' ' << (int)data[2] << ' ' << (int)data[3] << std::endl;

  trainLabels.read(data,4);
  labels.numLabels = (unsigned char)data[0]*0x1000000+(unsigned char)data[1]*10000+
                  (unsigned char)data[2]*0x100+(unsigned char)data[3];

  // std::cout << "\tNumber of Labels: " << labels.numLabels << std::endl;
  // std::cout << std::endl;

  images.buffer = new uint8_t[images.numImages*images.numRows*images.numCols];
  trainImages.read((char*)images.buffer,images.numImages*images.numRows*images.numCols);

  labels.buffer = new uint8_t[labels.numLabels];
  trainLabels.read((char*)labels.buffer,labels.numLabels);

  trainImages.close();
  trainLabels.close();
}
