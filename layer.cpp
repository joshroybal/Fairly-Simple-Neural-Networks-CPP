#include "layer.hpp"
#include "stats.hpp"

// Layer class private methods
void Layer::init_()
{
   float sigma = 1.0;
   int n = 0;
   if (previous_layer_) {
      n = previous_layer_->num_neurons_;
      // sigma = sqrt(2.0 / n);
      // sigma = sqrt(2.0 / (n + num_neurons_));
      sigma = sqrt(2.0) * sqrt(2.0 / (n + num_neurons_));
   }
   for (int i = 0; i < num_neurons_; i++) {
      if (previous_layer_) {
          dataSet x;
          // x.setNormal(n, 0.0, sigma);
          x.setTruncatedNormal(n, -2.0*sigma, 2.0*sigma, 0.0, sigma);
          std::vector<float> random_weights(x.getdataSet());
          // std::vector<float> random_weights(n);
          // std::vector<float>::iterator it;
          // for (it = random_weights.begin(); it != random_weights.end(); ++it)
          //    *it = rndint(0, 1) ? sigma_squared : 0.0;
         Neuron neuron(random_weights, learning_rate_);
         neurons_.push_back(neuron);            
      }
   }
   output_cache_.resize(num_neurons_);
   for (int i = 0; i < num_neurons_; i++)
      output_cache_[i] = 0.0; 
}

std::vector<float>& Layer::outputs_(const std::vector<float>& inputs, 
      float (*activation_function)(float))
{
   if (!previous_layer_)
      output_cache_ = inputs;
   else {
      output_cache_.resize(num_neurons_);
      for (int i = 0; i < num_neurons_; i++)
         output_cache_[i] = neurons_[i].Output(inputs, activation_function);
   }
   return output_cache_;
}

// should only be called on output layer
void Layer::calculate_deltas_for_output_layer_(const std::vector<float>& expected, float (*derivative_activation_function)(float))
{
    for (int n = 0; n < int(neurons_.size()); n++)
        neurons_[n].setDelta(derivative_activation_function(neurons_[n].getOutputCache()) * (expected[n] - output_cache_[n]));
}

// should not be called on output layer
void Layer::calculate_deltas_for_hidden_layer_(Layer& next_layer, 
      float (*derivative_activation_function)(float))
{
    for (int i = 0; i < num_neurons_; i++) {
        float sum = 0.0;
        for (int j = 0; j < int(next_layer.neurons_.size()); j++) {
            // next_layer neuron j's weight that connects to this neuron i
            sum += next_layer.neurons_[j].getWeight(i) * next_layer.neurons_[j].getDelta();
        }
        neurons_[i].setDelta(derivative_activation_function(neurons_[i].getOutputCache()) * sum);
    }
}
