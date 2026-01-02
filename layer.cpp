#include "layer.hpp"

std::vector<float>& Layer::outputs_(const std::vector<float>& inputs)
{
	if (!previous_layer_)
		output_cache_ = inputs;
	else {
		output_cache_.resize(num_neurons_);
		for (int i = 0; i < num_neurons_; i++)
			output_cache_[i] = neurons_[i].Output(inputs);
	}
	return output_cache_;
}

// should only be called on output layer
void Layer::calculate_deltas_for_output_layer_(const std::vector<float>& expected)
{
    for (int n = 0; n < int(neurons_.size()); n++)
        neurons_[n].setDelta(derivative_tanh(neurons_[n].getOutputCache()) * (expected[n] - output_cache_[n]));
}

// should not be called on output layer
void Layer::calculate_deltas_for_hidden_layer_(Layer& next_layer)
{
    for (int i = 0; i < num_neurons_; i++) {
        float sum = 0.0;
        for (int j = 0; j < int(next_layer.neurons_.size()); j++) {
            // next_layer neuron j's weight that connects to this neuron i
            sum += next_layer.neurons_[j].getWeight(i) * next_layer.neurons_[j].getDelta();
        }
        neurons_[i].setDelta(derivative_tanh(neurons_[i].getOutputCache()) * sum);
    }
}
