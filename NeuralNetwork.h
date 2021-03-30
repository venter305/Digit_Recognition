#pragma once

#include <Eigen/Dense>

//2 Layer NeuralNetwork
//input->hidden->output
class NeuralNetwork{
private:
  Eigen::MatrixXd weights_ih;
  Eigen::MatrixXd weights_ho;
  Eigen::VectorXd bias_h;
  Eigen::VectorXd bias_o;

  Eigen::VectorXd inputs;
  Eigen::VectorXd hidden;
  Eigen::VectorXd outputs;

  int numInputs;
  int numHidden;
  int numOutputs;

public:
  float learningRate = 0.5f;
  int batchSize = 10;
  float lambdaVal = 5.0f;

  NeuralNetwork(int numIn,int numHid,int numOut);

  void CalculateOutputs(float* input, float* nnOut = nullptr);
  void Train(int totalIterations, int batchSize, int numData,float*&nnIn, float* &answers, float learningRate, float lambda,bool test=false);
  float CalculateAccuracy(int numTestData,float* inputs, float* answers);

  int GetNumInputs(){return numInputs;}
  int GetNumHidden(){return numHidden;}
  int GetNumOutputs(){return numOutputs;}

  double* GetHidden(){return hidden.data();}

private:
  float Activation(float num);
  float ActivationDerivative(float num);

  float Softmax(float num, Eigen::VectorXd &weightSumMatrix);
  float CostDerivative(float output_activation, float target, float weightedSum);
};
