## AN APPROACH TO IMPROVE THE PERFORMANCE OF HPC APPLICATIONS USING NEURAL NETWORK-BASED APPROXIMATIONS

## Key Technologies, Processes and Deployment Methodologies:
• Python, Tensorflow (Python and C++), NumPy, Matplotlib, Windows 10, Unix (Ubuntu 16.04), CuDNN (GPU)

• Over 50 Neural Network model built, Profiling, Generation of Dataset, Training, Testing, Optimizations

• Agile Development, Object Oriented Design

• Container: Docker (CPU and GPU)


## Abstract
This project report describes an approach to enhance the performance of High-Performance Computing (HPC) applications like LAMMPS and how Neural Networks (NN) can be applied to run simulations. LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a large-scale scientific computing molecular dynamics application which requires a significant amount of computational resources to run large scale simulations. The Design, Architecture, Neural Network implementation approach, Results, and Conclusions are presented as applied to LAMMPS application. The results are very encouraging and motivating to apply Neural Networks implementation for such large-scale complex applications. Hence, Neural Networks implementation become very important for such approximation challenges via Machine Learning.
To achieve the above objectives, the time-consuming code blocks are evaluated to be replaced within the LAMMPS application with a deep Neural Network taking similar or less time in execution of the simulation whilst achieving same accuracy of the variables that are being used to run the simulation. The code region(s) to be replaced are based on two criteria, first being that the code regions must be time consuming and its execution time takes a large portion of total execution time of the application. Secondly, the replaced code region(s) should not have any impact on the correctness of the application considering a tolerable difference in variables that are used to run the simulation as Neural Networks are bound to induce marginal errors. Various different Neural Network models were built and tested in order to achieve similar accuracy as the original LAMMPS code.

## Deployment phase - Using Docker
TensorFlowCC is compiled using Bezel, and during compilation, it takes into account the underlying CPU or GPU on which it is going to run the code. To run the TensorflowCC code on CPU, a docker is required that has prebuilt files for running and compiling Tensorflow. A docker is a remotely hosted platform that helps to execute code from different platforms in a single container. The docker that is used in the project is hosted on a GitHub repository (Floopz GitHub repo.) [9]. This docker also offers GPU based version, on which TensorFlowCC can be built using CMake, although it needs a specific version of Nvidia Driver, Cuda toolkit, cuDNN. The same GitHub repository mentioned above has links to TensorFlowCC images of Docker. Running the GPU based Docker image of TensorFlowCC is not as straightforward as it might sound above. That Docker image needs to run on Nvidia Docker. NVidia Docker is also not very easy to work around with because it needs a GPU with CUDA compute-capability of 3.5 and above. To put in perspective, the Nvidia GTX Titan X has compute-capability of 3.5 but the GPU on my laptop has compute-capability of 3 or lower. A GPU with better compute capability was found that is mentioned above in section (2).

## The specification of the final Network that are used is as follows:
• 3 hidden layer Multi-Level Perceptron (MLP) built using TensorFlow.
• Input ->h1->Dropout->h2->h3-> Output (h: dense hidden layer)
• Activation in h1, h2 is ReLU. Activation in h3 (second last layer) is Tanh.
• 12 neurons/nodes in h1 and h3
• 20 neurons/nodes in h2
• Optimizer: Adam, learning rate = 0.0002
• Training samples: 75,000
• Validation samples: 25,000
• Test Loss: 0.01 (Mean Squared Error)

## Learning Phase of Neural Network:
Learning phase in a Multi-Layer Perceptron
Learning ability of the Neural Networks depend on the training method, amount of training data, the number of neurons and many other factors. The backpropagation algorithm [16] is a method for training the weights in a multilayer feed-forward Neural Network. As such, it requires a Network structure to be defined as one or more layers where one layer is fully connected to the next layer. The Backpropagation algorithm is a supervised learning method for multilayer feed-forward Networks from the field of Artificial Neural Networks. To put in simple terms, Backpropagation is like “learning from mistakes”. A Multi-Layer Perceptron consists of nodes in different layers; input layer, intermediate hidden layer(s) and the output layer. The connections between nodes of adjacent layers have “weights” associated with them. The goal of learning is to assign correct weights for these edges. Given an input vector, these weights determine what the output vector is. In supervised learning, the training set is labeled which means, for some given inputs (features), the actual output (label) is known. Initially, all the edge weights are randomly assigned. For every input in the training dataset, the Multi-Layer Perceptron is activated, and its output is observed. This predicted output is compared with the actual output, and the error is “propagated” back to the previous layer. This error is noted, and the weights are adjusted accordingly. This process is repeated until the output error is below a particular threshold. Once the above algorithm terminates, we have a “learned” model which, is considered ready to work with “new” inputs. This MLP is said to have learned from several examples (labeled data) and from its mistakes (error propagation). The total error at the output nodes is calculated and propagated backward through the Network using Backpropagation to calculate the gradients. The error at output neurons is passed to preceding layer neurons with the amount of the responsibility of each neuron for that error. The weights are updated by using the gradient which were backpropagated through the Network. Then an optimization method such as Gradient Descent [16] is used to adjust all weights in the Network with the aim of reducing the error at the output layer.

## Saving Weights and Graph to a Protobuf (.pb) file
Trying different Networks in TensorFlow’s python API helps in productivity. Once the final design of the Network is decided, the weights and the graph can be stored in a protocol buffer file (.pb file). This protocol buffer file can be then imported into the TensorFlow’s C++ API, which can be integrated with the LAMMPS source code [10]. The .pb file can be created in TensorFlow’s python API using the TensorFlow’s freeze_graph() function. This function takes input as Checkpoints, its files and the Graph before training. The .pb file is very important from the standpoint of portability of the Deep Learning model to different CPU architectures, Operating Systems and Programming environments. In this project, the .pb file generated using the Tensorflow Python API is imported in to Tensorflow CPP environment.
## Results:
The main goal of this project was to decrease the time taken by the specific code block that was selected to be replaced. Following the dataflow diagram in Figure 6, a similar time is achieved after replacing the C++ code block with Deep Neural Network. On CPU, 1 iteration of heat exchange simulation took 0.001837 seconds to be executed. When this number is multiped by the total number of simulations, the time obtained is similar to time taken by the actual code block. Figure 9 and 10 shows the results on CPU and GPU respectively.
Using cuDNN it was observed that the time was further reduced by 10 times. It took only 0.00028 seconds for 1 iteration to execute. It can be clearly seen that this is faster than the original LAMMPS code.

## Conclusion
In this project, a Neural Network-based approximation approach to improve the performance of a high-performance computing application is implemented. As Neural Networks have the ability to generalize well and respond to unexpected input patterns, it is a very productive idea to replace the code blocks with highly optimized Neural Networks within any type of high-performance computing application. In the title of the project, the word “approximations” justify the fact that Neural Network are bound to induce some level of errors in the computations. The most time-consuming code block within LAMMPS application is selected and replaced with a Neural Network. The built Network is designed keeping two things in mind. Firstly, it should take similar or less time in the executing a particular block of code and secondly, it should not have any impact in the correctness of the application. The results section provides details about the correctness of the application. It is seen that almost all the predicted values are in close proximity to the actual values. The execution time is also similar to the original code on CPU. When the Neural Network is executed using CuDNN, it is seen it takes much less time to execute the code block that was selected to be replaced.

## Code:
### Data Cleaning for Neural Network folder: 
Preprocessing was done to the generated datasets. code for data cleaning can be found in this folder.
### Modified files in Lammps:  
As the Neural networks code is plugged into LAMMPS application, some source files have been modified to meet the Tensorflow dependencies.
### Tensorflow Neural Network code: 
Actual Neural Network model
### TensorFlowCC code for LAMMPS: 
As LAMMPS is written in C++, a protobuf file was generated using TensorFlow Python API and plugged into LAMMPS C++ code using TenorfFlow C++ API.
