#pragma once

class MNISTDataLoader{
  public:
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

  public:
    static void LoadData(MNISTImages &images, MNISTLabels &labels){

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

};
