#include <iostream>
#include "network.hpp"

void Network::init_(const std::vector<int>& layer_structure, float learning_rate)
{
    if (layer_structure.size() != 3) return;
    layers_.resize(layer_structure.size());
    // input layer
    layers_[0] = input_layer_;
    // hidden layers and output layer
    int n = layers_.size();
    for (int i = 1; i < n; i++) {
		Layer next_layer(&layers_[i - 1], layer_structure[i], learning_rate);
		layers_[i] = next_layer;
	}
}

// Pushes input data to the first layer, then output from the first
// as input to the second, second to the third, etc.
std::vector<float> Network::outputs_(const std::vector<float>& input,
      float (*activation_function)(float))
{
    std::vector<float> output = input;
    std::vector<Layer>::iterator it;
    for (it = layers_.begin(); it != layers_.end(); it++)
        output = it->Outputs(output, activation_function);
    return output;
}

// Figure out each neuron's changes based on the errors of the output
// versus the expected outcome.
void Network::backpropagate_(const std::vector<float>& expected, 
      float (*derivative_activation_function)(float))
{
	// calculate delta for output layer neurons
	int last_layer = layers_.size() - 1;
	layers_[last_layer].calculateDeltasForOutputLayer(expected, derivative_activation_function);
	// calculate deltas for hidden layers in reverse order
	for (int i = last_layer - 1; i > 0; i--)
		layers_[i].calculateDeltasForHiddenLayer(layers_[i + 1], derivative_activation_function);
}

// backpropogate_ doesn't actually change any weights
// this method uses the deltas calculated by backpropagate_ to
// actually make changes to the weights.
void Network::update_weights_()
{
    int n = layers_.size();
	for (int i = 1; i < n; i++)	{	// skip input layer
		Layer& layer = layers_[i];
		for (int j = 0; j < layer.getNumNeurons(); j++) {
			Neuron& neuron = layer.getNeuron(j);
			for (int k = 0; k < neuron.getNumWeights(); k++) {
				neuron.setWeight(k, neuron.getWeight(k) + (neuron.getLearningRate()
				* (layer.getPreviousLayer()->getOutputCache()[k]) * neuron.getDelta()));
			}
		}
	}
}

void Network::dump_()
{
    int n = layers_.size();
    for (int i = 0; i < n; i++) {
        std::cout << "layer: " << i << std::endl;
        print_vector(layers_[i].getOutputCache());
        if (i > 0)
            for (int j = 0; j < layers_[i].getNumNeurons(); j++)
                layers_[i].getNeuron(j).printWeights();
    }
}

void Network::train_(std::vector<std::vector<float> >& inputs,
        std::vector<std::vector<float> >& expecteds,
        float (*activation_function)(float),
        float (*derivative_activation_function)(float))
{
    int n = inputs.size();
    for (int i = 0; i < n; i++) {
        std::vector<float> ys(expecteds[i]);
        std::vector<float> outs(outputs_(inputs[i], activation_function));
        backpropagate_(ys, derivative_activation_function);
        update_weights_();
    }
}

int Network::validate_(std::vector<std::vector<float> >& inputs,
        std::vector<std::string>& expecteds,
        const std::vector<std::string>& headers,
        std::string (*interpret_output)(const std::vector<float>&, 
						const std::vector<std::string>&),
        float (*activation_function)(float))
{
    int correct = 0;
    int n = inputs.size();
    for (int i = 0; i < n; i++) {
        std::string result = interpret_output(outputs_(inputs[i], activation_function), headers);
        if (result == expecteds[i])
            ++correct;
    }
    return correct;
}
