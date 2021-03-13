#include <iostream>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int numIn,int numHid,int numOut){
  numInputs = numIn;
  numHidden = numHid;
  numOutputs = numOut;

  weights_ih.resize(numHidden,numInputs);
  for (int i=0;i<numHidden;i++){
    for (int j=0;j<numInputs;j++)
      //weights_ih(i,j) = 1.0f;
      weights_ih(i,j) = ((float)rand()/(float)RAND_MAX)*2-1;
  }

  weights_ho.resize(numOutputs,numHidden);
  for (int i=0;i<numOutputs;i++){
    for (int j=0;j<numHidden;j++){
      //weights_ho(i,j) = 1.0f;
      weights_ho(i,j) = ((float)rand()/(float)RAND_MAX)*2-1;
    }
  }

  bias_h.resize(numHidden);
  for (int i=0;i<numHidden;i++)
    //bias_h(i) = 1.0f;
    bias_h(i) = ((float)rand()/(float)RAND_MAX)*2-1;

  bias_o.resize(numOutputs);
  for (int i=0;i<numOutputs;i++)
    //bias_o(i) = 1.0f;
    bias_o(i) = ((float)rand()/(float)RAND_MAX)*2-1;

  inputs.resize(numInputs);
  hidden.resize(numHidden);
  outputs.resize(numOutputs);

}

void NeuralNetwork::CalculateOutputs(float* input, float* nnOut){

  for (int i=0;i<numInputs;i++){
    inputs(i) = input[i];
  }

  //Calculate Hidden Layer

  for (int i=0;i<numHidden;i++){
    auto weightSumMatrix = weights_ih*inputs+bias_h;
    float weightedSum = weightSumMatrix(i,0);
    hidden(i) = Activation(weightedSum);
  }
  //Calculate Output Layer

  for (int i=0;i<numOutputs;i++){
    auto weightSumMatrix = weights_ho*hidden+bias_o;
    float weightedSum = weightSumMatrix(i,0);
    outputs(i) = Activation(weightedSum);
    if(nnOut != nullptr)
      nnOut[i] = outputs(i);
  }

}

void NeuralNetwork::Train(int totalIterations, int batchSize, int numData,float* &nnIn, float* &answers, float learningRate,bool test){
  for (int itCount=0;itCount<totalIterations;itCount++){

    Eigen::MatrixXd batchWeights_ih(numHidden,numInputs);
    Eigen::MatrixXd batchWeights_ho(numOutputs,numHidden);
    Eigen::VectorXd batchBias_h(numHidden);
    Eigen::VectorXd batchBias_o(numOutputs);

    int *indexArray = new int[numData];
    for(int i=0;i<numData;i++)
      indexArray[i] = i;
    std::random_shuffle(indexArray,indexArray+numData);

    int numBatches = numData/batchSize;

    for (int currBatch=0;currBatch < numBatches;currBatch++){

      batchWeights_ih *= 0;
      batchWeights_ho *= 0;
      batchBias_h *= 0;
      batchBias_o *= 0;

      for (int bCount=0;bCount<batchSize;bCount++){
        std::cout << '\r' << "Epoch: " << itCount << " Current Batch: " << currBatch << " Batch Count: " << bCount << std::flush;
        //int i = rand()%numData;

        int index = indexArray[currBatch*batchSize+bCount];

        CalculateOutputs(nnIn+index*numInputs);

        //Error for Last Layer

        //Cost Gradient
        Eigen::VectorXf costGradient_o(numOutputs);
        //Error Value
        Eigen::VectorXd outputError(numOutputs);
        for (int j=0;j<numOutputs;j++){
          costGradient_o(j) = CostDerivative(outputs(j),answers[index*numOutputs+j]);
          outputError(j) =   costGradient_o(j) * (outputs(j)*(1-outputs(j)));
        }


        //Hidden Layer

        Eigen::VectorXd hiddenError(numHidden);
        Eigen::VectorXd previousError = weights_ho.transpose()*outputError;

        //hiddenError = previousError.cwiseProduct(hidden.cwiseProduct(Eigen::VectorXd::Ones(numHidden)-hidden));

        for (int j=0;j<numHidden;j++){
          hiddenError(j) = previousError(j) * (hidden(j)*(1-hidden(j)));
        }

        batchWeights_ho += (outputError * hidden.transpose());

        batchWeights_ih += (hiddenError * inputs.transpose());

        batchBias_h += hiddenError;
        batchBias_o += outputError;

      }

      weights_ho -= learningRate/batchSize * batchWeights_ho;

      weights_ih -= learningRate/batchSize * batchWeights_ih;

      bias_h -= learningRate/batchSize * batchBias_h;
      bias_o -= learningRate/batchSize * batchBias_o;

    }

    if (test)std::cout << " Accuracy: " << CalculateAccuracy(100,nnIn,answers) << '%' << std::endl;

    delete[] indexArray;
    // delete[] inputArray;
    // delete[] targetArray;
  }
}

float NeuralNetwork::CalculateAccuracy(int numTestData,float* inputs, float* answers){
    int sum = 0;
    float *out = new float[numOutputs];
    for(int i=0;i<numTestData;i++){
      CalculateOutputs(&inputs[i*numInputs],out);
      float max = 0;
      int maxIndex = 0;
      int target = 0;
      for (int j=0;j<numOutputs;j++){
        if (answers[i*numOutputs+j] == 1)
          target = j;
        if ( out[j] >= max){
          max = out[j];
          maxIndex = j;
        }
      }

      if (maxIndex == target)
        sum++;
    }

    delete[] out;

    return (float)sum/(float)numTestData;
}

float NeuralNetwork::Activation(float num){

  float sum = 1.0f/(1.0f+std::exp(num*-1));
  return sum;
}

float NeuralNetwork::CostDerivative(float output_activation, float target){
  return output_activation-target;
}
