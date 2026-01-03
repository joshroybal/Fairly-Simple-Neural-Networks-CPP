#include <iostream>
#include <cstdlib>
#include "stats.hpp"
#include "util.hpp"

static const float PI = 3.1415926539F;

float dataSet::random_() const
{
    float z = rand() / float(RAND_MAX);
    while (z == 0)
        z = rand() / float(RAND_MAX);
    return z;
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
    float sigma = 0.9973F / 3.0F;
    x_.clear();
    x_.resize(n);
    for (int i = 0; i < n; i += 2)
        while (true) {
            float u = random_();
            float v = random_();
            float r = sqrt(-2.0F * log(u));
            float z1 = sigma * r * cos(2.0F * PI * v);
            float z2 = sigma * r * sin(2.0F * PI * v);
            if (z1 >= -1.0F && z1 <= 1.0F && z2 >= -1.0F && z2 <= 1.0F) {
                x_[i] = z1;
                if ((i + 1) < n) x_[i + 1] = z2;
                break;
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
