#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <string>

float uniform();
float normal();
float dot_product(const std::vector<float>&, const std::vector<float>&);
float derivative_tanh(float);
void normalize_by_feature_scaling(std::vector<std::vector<float> >&);
void print_vector(const std::vector<int>&);
void print_vector(const std::vector<float>&);
void print_vector(const std::vector<std::string>&);

#endif
