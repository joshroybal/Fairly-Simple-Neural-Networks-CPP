#include <cmath>
#include "neuron.hpp"

float Neuron::output_(const std::vector<float>& inputs)
{
   output_cache_ = dot_product(inputs, weights_);
   return tanh(output_cache_);
}
