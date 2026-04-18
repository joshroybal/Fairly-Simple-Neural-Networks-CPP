#include <cmath>
#include "neuron.hpp"
#include "util.hpp"

float Neuron::output_(const std::vector<float>& inputs, 
      float (*activation_function)(float))
{
   output_cache_ = dot_product(inputs, weights_);
   return activation_function(output_cache_);
}
