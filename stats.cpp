#include <iostream>
#include <cstdlib>
#include <cfloat>
#include "stats.hpp"

static const int PI = 3.1415926539;

float dataSet::random_() const
{
    return rand() / (float(RAND_MAX) + 2e-36);
}

void dataSet::set_uniform_(int n)
{
    x_.clear();
    x_.resize(n);
    for (std::vector<float>::iterator it = x_.begin(); it != x_.end(); ++it)
        *it = random_();
}

void dataSet::set_normal_(int n, float mu, float sigma)
{
    x_.clear();
    x_.resize(n);
    for (std::vector<float>::iterator it = x_.begin(); it != x_.end(); ++it) {
        if ((it - x_.begin()) % 2)
            *it = mu + (sigma * sqrt(-2*log(random_())) * cos(2*PI*random_()));
        else
            *it = mu + (sigma * sqrt(-2*log(random_())) * sin(2*PI*random_()));
    }
}

void dataSet::set_scaled_normal_(int n)
{
    float mu = 0.0;
    float sigma = 0.997 / 3.0;
    x_.clear();
    x_.resize(n);
    for (std::vector<float>::iterator it = x_.begin(); it != x_.end(); ++it) {
        if ((it - x_.begin()) % 2) {
            *it = mu + (sigma * sqrt(-2*log(random_())) * cos(2*PI*random_()));
            while (*it <= -1.0 || *it >= 1.0)
                *it = mu + (sigma * sqrt(-2*log(random_())) * cos(2*PI*random_()));

        }
        else {
            *it = mu + (sigma * sqrt(-2*log(random_())) * sin(2*PI*random_()));
            while (*it <= -1.0 || *it >= 1.0)
                *it = mu + (sigma * sqrt(-2*log(random_())) * cos(2*PI*random_()));

        }
    }
}

void dataSet::print_dataSet_() const
{
    std::vector<float>::const_iterator cit;
    for (cit = x_.begin(); cit != x_.end(); cit++) std::cout << *cit << std::endl;
}

float dataSet::mean_() const
{
    float acc = 0.0;
    std::vector<float>::const_iterator cit;
    for (cit = x_.begin(); cit != x_.end(); ++cit)
        acc += *cit;
    acc /= x_.size();
    return acc;
}

float dataSet::var_(float mu) const
{
    float acc = 0.0;
    std::vector<float>::const_iterator cit;
    for (cit = x_.begin(); cit != x_.end(); ++cit)
        acc += (*cit - mu) * (*cit - mu);
    acc /= (x_.size() - 1.0);
    return acc;
}
