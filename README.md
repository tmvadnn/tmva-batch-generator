# tmva-batch-generator
Prototype for the GSoC'22 project ROOT - Machine Learning Developments - Batch Generator for training machine learning models.


## Project Description 
Toolkit for Multivariate Analysis (TMVA) is a multi-purpose machine learning toolkit integrated into the ROOT scientific software framework, used in many particle physics data analysis and applications. Since it is part of the ROOT data analysis framework, it comes with an automatically generated Python interface, which closely follows the C++ interface. The goal of this project is to develop a generator in C++ and Python to read data from the ROOT I/O and input them to the Python machine learning tools such as Tensorflow/Keras and PyTorch. The main aim of the generator is to efficiently input data from the ROOT I/O system to train machine learning models, and keep in memory only the data required to train a batch of events and not all the data set. 

## Goals 
1. Development of a Python generator for getting the data directly from a ROOT TTree using PyROOT interfaces with Documentation 
2. Integration with ROOT RDataFrame with Documentation 
3. Development of tests and tutorial example
