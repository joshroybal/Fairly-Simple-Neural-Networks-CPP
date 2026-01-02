#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include "util.hpp"

class Neuron {
   public:
      Neuron(std::vector<float>& weights, float learning_rate) :
         weights_(weights), learning_rate_(learning_rate), output_cache_(0.0),
         delta_(0.0) {};
      int getNumWeights() { return weights_.size(); };
      float getWeight(int n) { return weights_[n]; };
      float getLearningRate() { return learning_rate_; };
      float getOutputCache() { return output_cache_; };
      float getDelta() { return delta_; };
      float Output(const std::vector<float>& inputs) { return output_(inputs); };
      void setWeight(int n, float r) { weights_[n] = r; }
      void setDelta(float n) { delta_ = n; };
      void printWeights() { print_vector(weights_); };
   private:
      std::vector<float> weights_;
      float learning_rate_;
      float output_cache_;
      float delta_;
      float output_(const std::vector<float>&);
};

#endif
