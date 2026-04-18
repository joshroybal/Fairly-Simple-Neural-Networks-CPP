#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <string>

float dot_product(const std::vector<float>&, const std::vector<float>&);
float sigmoid(float);
float derivative_sigmoid(float);
float derivative_tanh(float);
float ReLU(float);
float derivative_ReLU(float);
float leaky_ReLU(float);
float derivative_leaky_ReLU(float);
void normalize_by_feature_scaling(std::vector<std::vector<float> >&);
void normalize_by_z_score(std::vector<std::vector<float> >&);
void print_vector(const std::vector<int>&);
void print_vector(const std::vector<float>&);
void print_vector(const std::vector<std::string>&);

#endif
