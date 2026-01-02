#ifndef LAYER_HPP
#define LAYER_HPP

#include "neuron.hpp"
#include "stats.hpp"
//#include "util.hpp"

class Layer {
   public:
      Layer() {};
      Layer(Layer *previous_layer, int num_neurons, float learning_rate) :
         previous_layer_(previous_layer), num_neurons_(num_neurons),
         learning_rate_(learning_rate)
      {
         for (int i = 0; i < num_neurons_; i++) {
            std::vector<float> random_weights;
            if (previous_layer_) {
				int n = previous_layer_->num_neurons_;
                dataSet x;
                x.setscaledNormal(n);
                random_weights = x.getdataSet();
                Neuron neuron(random_weights, learning_rate_);
                neurons_.push_back(neuron);
            }
         }
         output_cache_.resize(num_neurons_);
         for (int i = 0; i < num_neurons_; i++)
             output_cache_[i] = 0.0;
      };
      Layer* getPreviousLayer() { return previous_layer_; };
      int getNumNeurons() { return num_neurons_; };
      Neuron& getNeuron(int n) { return neurons_[n]; };
      std::vector<float>& Outputs(const std::vector<float>& inputs)
      { return outputs_(inputs); };
      std::vector<float>& getOutputCache() { return output_cache_; };
      void calculateDeltasForOutputLayer(const std::vector<float>& expected)
      { calculate_deltas_for_output_layer_(expected); };
      void calculateDeltasForHiddenLayer(Layer& next_layer)
      { calculate_deltas_for_hidden_layer_(next_layer); };
   private:
      Layer *previous_layer_;
      int num_neurons_;
      float learning_rate_;
      std::vector<Neuron> neurons_;
      std::vector<float>& outputs_(const std::vector<float>&);
      std::vector<float> output_cache_;
      void calculate_deltas_for_output_layer_(const std::vector<float>&);
      void calculate_deltas_for_hidden_layer_(Layer&);
};

#endif
