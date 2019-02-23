#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"

#include <ctime>
#include <cmath> 

using namespace std;
int main(int argc, char** argv) {
   

std::string PathGraph = "./new_frozen_graph.pb";

//Setup Input Tensors 
//	tensorflow::Tensor Input1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,9}));

	tensorflow::Tensor Input0(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,9}));
	
	auto input_tensor_mapped = Input0.tensor<float, 2>();
   	 double c2=0;
    	 double c3=0;
   	 double c1= clock();

	// set the (4,2) possible input values for XOR
	input_tensor_mapped(0, 0) = 1.73297;
	input_tensor_mapped(0, 1) = 48;
	input_tensor_mapped(0, 2) = 24;
	input_tensor_mapped(0, 3) = -0.0109438;
	input_tensor_mapped(0, 4) = 1;
	input_tensor_mapped(0, 5) = -1.31;
	input_tensor_mapped(0, 6) = -0.0763915;
	input_tensor_mapped(0, 7) = -0.105023;
	input_tensor_mapped(0, 8) = 3;

	
	// Output
	std::vector<tensorflow::Tensor> output;

	//initial declaration Tensorflow
	tensorflow::Session* session;
	tensorflow::Status status;
	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
   		std::cout << status.ToString() << "\n";
    	return 1;
    }
    // Define Graph
	tensorflow::GraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);
	
	if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
     	return 1;
   	}

   	// Add the graph to the session
  	status = session->Create(graph_def);
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
        return 1;
    }
   c2=clock();
    // Feed dict
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
   		 { "IteratorGetNext:0", Input0},
    	};
		status = session->Run(inputs, {"prediction/BiasAdd:0"},{}, &output);
		if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
   		return 1;
  		}

		auto Result = output[0].matrix<float>();
		std::cout << "Actual Values:  -------------------------->  2.13543    0.124526    0.171197    -2.13543    -0.124526    -0.171197   0"<<endl;
		std::cout << "IteratorGetNext:0 | prediction/BiasAdd:0: "<< Result << std::endl;
   c3=clock();

	c1=c1/CLOCKS_PER_SEC;
	c2=c2/CLOCKS_PER_SEC;
	c3=c3/CLOCKS_PER_SEC;

	cout<<"\nTime difference (seconds)--------------------------------->"<<c3-c1<<endl;
	cout<<"\nTime difference (seconds) before session ----------------->"<<c3-c2<<endl;
	/**
	inputs = {
   		 { "Input:0", Input1},
    	};
		status = session->Run(inputs, {"Layer2/Output"},{}, &output);
		if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
   		return 1;
  		}
		auto Result1 = output[0].matrix<float>();
		std::cout << "Input: 1 | Output: "<< Result1(0,0) << std::endl;	
	**/	
	
}
