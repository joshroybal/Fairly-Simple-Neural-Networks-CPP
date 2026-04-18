#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "util.hpp"
#include "stats.hpp"

float sigmoid(float x)
{
   return 1.0 / (1.0 + exp(-x));
}

float derivative_sigmoid(float x)
{
   float sig = sigmoid(x);
   return sig * (1.0 - sig);
}

float derivative_tanh(float x)
{
   float y = tanh(x);
   return 1.0 - y * y;
}

float ReLU(float x)
{
   return (x > 0) ? x : 0.0;
}

float derivative_ReLU(float x)
{
   return (x > 0) ? 1.0 : 0.0;
}

float leaky_ReLU(float x)
{
   return (x > 0) ? x : 0.01 * x;
}

float derivative_leaky_ReLU(float x)
{
   return (x > 0) ? 1.0 : 0.01;
}

float dot_product(const std::vector<float>& u, const std::vector<float>& v)
{
    float res = 0.0;
    int n = std::min(u.size(), v.size());
    for (int i = 0; i < n; i++) res += u[i] * v[i];
    return res;
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
            dataset[row_num][col_num] = 0.25 * (dataset[row_num][col_num] - *minimum) / (*maximum - *minimum);
    }
}

void normalize_by_z_score(std::vector<std::vector<float> >& dataset)
{
    int m = dataset.size();
    int n = dataset[0].size();
    for (int j = 0; j < n; ++j) {
        std::vector<float> column(m);
        for (int i = 0; i < m; ++i)
            column[i] = dataset[i][j];
        dataSet feature(column);
        std::vector<float> z(feature.getZScores());
        for (int i = 0; i < m; ++i)
            dataset[i][j] = z[i];
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
    for (std::vector<float>::const_iterator cit = v.begin(); cit != v.end(); cit++) {
        std::cout << *cit;
        if (cit + 1 != v.end()) std::cout << ',';
    }
    std::cout << std::endl;
}

void print_vector(const std::vector<std::string>& v)
{
    std::vector<std::string>::const_iterator cit;
    for (cit = v.begin(); cit != v.end(); cit++)
        std::cout << *cit << std::endl;
}
