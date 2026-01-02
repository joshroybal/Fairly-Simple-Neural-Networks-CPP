#include <iostream>
#include <cmath>
#include <algorithm>
#include "util.hpp"

float uniform()
{
    return rand() / float(RAND_MAX);
}

float normal()
{
    static int i = 0;
    const float pi = 3.14159265359;
    const float sigma = 0.997 / 3.0;
    ++i;
    while (true) {
        float u = uniform();
        float v = uniform();
        float z;
        if (i % 2 == 0) 
            z = sigma * sqrt(-2.0 * log(u + 1e-9)) * cos(2.0 * pi * v);
        else
            z = sigma * sqrt(-2.0 * log(u + 1e-9)) * sin(2.0 * pi * v);
        if (z > -1.0 && z < 1.0) return z;
    }
}

float dot_product(const std::vector<float>& u, const std::vector<float>& v)
{
    float res = 0.0;
    int n = std::min(u.size(), v.size());
    for (int i = 0; i < n; i++) res += u[i] * v[i];
    return res;
}

float derivative_tanh(float x)
{
    float t = tanh(x);
    return 1 - t*t;
}

void normalize_by_feature_scaling(std::vector<std::vector<float> >& dataset)
{
    int n = dataset[0].size();
    int numrows = dataset.size();
    for (int col_num = 0; col_num < n; col_num++) {
        std::vector<float> column;
        for (int row_num = 0; row_num < numrows; row_num++)
            column.push_back(dataset[row_num][col_num]);
        std::vector<float>::const_iterator maximum;
        maximum = std::max_element(column.begin(), column.end());
        std::vector<float>::const_iterator minimum;
        minimum = std::min_element(column.begin(), column.end());
        for (int row_num = 0; row_num < numrows; row_num++)
            dataset[row_num][col_num] = 2 * ((dataset[row_num][col_num] - *minimum) / (*maximum - *minimum)) - 1;
    }
}

void print_vector(const std::vector<int>& v)
{
   for (std::vector<int>::const_iterator cit = v.begin(); cit != v.end(); cit++)
      std::cout << ' ' << *cit;
   std::cout << std::endl;
}

void print_vector(const std::vector<float>& v)
{
    for (std::vector<float>::const_iterator cit = v.begin(); cit != v.end(); cit++)
        std::cout << ' ' << *cit;
    std::cout << std::endl;
}

void print_vector(const std::vector<std::string>& v)
{
    std::vector<std::string>::const_iterator cit;
    for (cit = v.begin(); cit != v.end(); cit++)
        std::cout << *cit << std::endl;
}
