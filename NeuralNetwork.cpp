#include <iostream>
#include <iomanip>
#include <random>
#include <time.h>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int numIn,int numHid,int numOut){
  numInputs = numIn;
  numHidden = numHid;
  numOutputs = numOut;

  std::default_random_engine generator(time(NULL));
  std::normal_distribution<double> randomNum(0,1.0f/std::sqrt(numInputs));

  weights_ih.resize(numHidden,numInputs);
  for (int i=0;i<numHidden;i++){
    for (int j=0;j<numInputs;j++)
      weights_ih(i,j) = randomNum(generator);
  }

  weights_ho.resize(numOutputs,numHidden);
  for (int i=0;i<numOutputs;i++){
    for (int j=0;j<numHidden;j++){
      weights_ho(i,j) = randomNum(generator);
    }
  }

  bias_h.resize(numHidden);
  for (int i=0;i<numHidden;i++)
    bias_h(i) = randomNum(generator);

  bias_o.resize(numOutputs);
  for (int i=0;i<numOutputs;i++)
    bias_o(i) = randomNum(generator);

  inputs.resize(numInputs);
  hidden.resize(numHidden);
  outputs.resize(numOutputs);

}

void NeuralNetwork::CalculateOutputs(float* input, float* nnOut){

  for (int i=0;i<numInputs;i++){
    inputs(i) = input[i];
  }

  //Calculate Hidden Layer

  Eigen::VectorXd weightSumMatrix_H = weights_ih*inputs+bias_h;
  for (int i=0;i<numHidden;i++){
    float weightedSum = weightSumMatrix_H(i);
    hidden(i) = Activation(weightedSum);
  }

  //Calculate Output Layer

  Eigen::VectorXd weightSumMatrix_O = weights_ho*hidden+bias_o;
  for (int i=0;i<numOutputs;i++){
    float weightedSum = weightSumMatrix_O(i);
    outputs(i) = Softmax(weightedSum,weightSumMatrix_O);
    if(nnOut != nullptr)
      nnOut[i] = outputs(i);
  }

}

void NeuralNetwork::Train(int totalIterations, int batchSize, int numData,float* &nnIn, float* &answers, float learningRate, float lambda,bool test){
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

      batchWeights_ih.setZero();
      batchWeights_ho.setZero();
      batchBias_h.setZero();
      batchBias_o.setZero();

      for (int bCount=0;bCount<batchSize;bCount++){
        std::cout << '\r' << "Epoch: " << itCount << " Current Batch: " << currBatch << " Batch Count: " << bCount << std::flush;
        //int i = rand()%numData;

        int index = indexArray[currBatch*batchSize+bCount];

        CalculateOutputs(&nnIn[index*numInputs]);

        //Error for Last Layer

        //Cost Gradient
        Eigen::VectorXf costGradient_o(numOutputs);
        //Error Value
        Eigen::VectorXd outputError(numOutputs);
        Eigen::VectorXd weightSumMatrix_O = weights_ho*hidden+bias_o;
        for (int j=0;j<numOutputs;j++){
          outputError(j) = CostDerivative(outputs(j),answers[index*numOutputs+j],weightSumMatrix_O(j));// * ActivationDerivative(weightSumMatrix_O(j));
        }


        //Hidden Layer

        Eigen::VectorXd hiddenError(numHidden);
        Eigen::VectorXd previousError = weights_ho.transpose()*outputError;

        //hiddenError = previousError.cwiseProduct(hidden.cwiseProduct(Eigen::VectorXd::Ones(numHidden)-hidden));

        Eigen::VectorXd weightSumMatrix_H = weights_ih*inputs+bias_h;
        for (int j=0;j<numHidden;j++){
          hiddenError(j) = previousError(j) * ActivationDerivative(weightSumMatrix_H(j));
        }

        batchWeights_ho += (outputError * hidden.transpose());

        batchWeights_ih += (hiddenError * inputs.transpose());

        batchBias_h += hiddenError;
        batchBias_o += outputError;

      }

      weights_ho = (1-(learningRate*lambda)/numData)*weights_ho - learningRate/batchSize * batchWeights_ho;

      weights_ih = (1-(learningRate*lambda)/numData)*weights_ih - learningRate/batchSize * batchWeights_ih;

      bias_h -= learningRate/batchSize * batchBias_h;
      bias_o -= learningRate/batchSize * batchBias_o;

    }

    if (test)std::cout << " Accuracy: " << CalculateAccuracy(100,nnIn,answers)*100 << '%';
    std::cout << std::endl;

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

//Sigmoid Function
float NeuralNetwork::Activation(float num){

  float sum = 1.0f/(1.0f+std::exp(num*-1));
  return sum;
}

float NeuralNetwork::ActivationDerivative(float num){
  return Activation(num) * (1-Activation(num));
}

float NeuralNetwork::Softmax(float num, Eigen::VectorXd &weightSumMatrix){
  float sum = 0;
  for (int i = 0;i<numOutputs;i++)
    sum += std::exp(weightSumMatrix(i));

  float out = std::exp(num)/sum;
  return out;
}

//Quadratic Cost
//1/2*(target-output)^2
// float NeuralNetwork::CostDerivative(float output_activation, float target, float weightedSum){
//   return (output_activation-target) * ActivationDerivative(weightedSum);
// }

//Cross-Entropy Cost
//sum(-target*ln(a)-(1-y)*ln(1-a))
float NeuralNetwork::CostDerivative(float output_activation, float target, float weightedSum){
  return (output_activation-target);
}
