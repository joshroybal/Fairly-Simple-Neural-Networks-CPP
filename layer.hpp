#ifndef LAYER_HPP
#define LAYER_HPP

#include "neuron.hpp"
#include "stats.hpp"    
         
class Layer {
   public:
      Layer() {};
      Layer(Layer *previous_layer, int num_neurons, float learning_rate) :
         previous_layer_(previous_layer), num_neurons_(num_neurons),
         learning_rate_(learning_rate) { init_(); };
      Layer* getPreviousLayer() { return previous_layer_; };
      int getNumNeurons() { return num_neurons_; };
      Neuron& getNeuron(int n) { return neurons_[n]; };
      std::vector<float>& Outputs(const std::vector<float>& inputs,
              float (*activation_function)(float))
      { return outputs_(inputs, activation_function); };
      std::vector<float>& getOutputCache() { return output_cache_; };
      void calculateDeltasForOutputLayer(const std::vector<float>& expected,
              float (*derivative_activation_function)(float))
      { calculate_deltas_for_output_layer_(expected, derivative_activation_function); };
      void calculateDeltasForHiddenLayer(Layer& next_layer,
              float (*derivative_activation_function)(float))
      { calculate_deltas_for_hidden_layer_(next_layer, derivative_activation_function); };
   private:
      void init_();
      Layer *previous_layer_;
      int num_neurons_;
      float learning_rate_;
      std::vector<Neuron> neurons_;
      std::vector<float>& outputs_(const std::vector<float>&, 
              float (*)(float));
      std::vector<float> output_cache_;
      void calculate_deltas_for_output_layer_(const std::vector<float>&, 
              float (*)(float));
      void calculate_deltas_for_hidden_layer_(Layer&, float (*)(float));
};

#endif
