#include <iostream>
#include <cstdlib>
#include "stats.hpp"
#include "util.hpp"

static const float PI = 3.1415926539F;

// subroutines
float qsel(const std::vector<float>& x, int n)
{
	int l = 0;
	int r = x.size() - 1;
	std::vector<int> idx(x.size());
	for (int i = 0; i < int(idx.size()); ++i)
		idx[i] = i;
	while (l <= r) {
		int pividx = part(x, idx, l, r);
		if (n < pividx)
			r = pividx - 1;
		else if (n > pividx)
			l = pividx + 1;
		else
			break;
	}
	return x[idx[n]];
}

int part(const std::vector<float>& x, std::vector<int>& idx, int lo, int hi)
{
	float pividx = rndint(lo, hi);
	float pivot = x[idx[pividx]];
	swap(idx[pividx], idx[hi]);
	int stored_index = lo;
	for (int i = lo; i < hi; ++i)
		if (x[idx[i]] < pivot) {
			swap(idx[stored_index], idx[i]);
			++stored_index;
		}
	swap(idx[hi], idx[stored_index]);
	return stored_index;
}

int rndint(int lo, int hi)
{
	return lo + int((hi - lo + 1) * (rand() / (float(RAND_MAX) + 1.0)));
}

void swap(int& m, int& n)
{
	int tmp = m;
	m = n;
	n = tmp;
}

// dataSet class private methods
std::vector<float> dataSet::z_scores_() const
{
    float mu = mean_();
    float sigma = sqrt(var_(mu));
    std::vector<float> z(x_.size());
    for (int i = 0; i < int(x_.size()); ++i)
        z[i] = (x_[i] - mu) / sigma;
    return z;
}

std::vector<float> dataSet::get_stats_() const
{
    float mu = mean_();
    float nu = median_(); 
    std::vector<float> stats;
    stats.push_back(mu);
    stats.push_back(sqrt(var_(mu)));
    stats.push_back(nu);
    stats.push_back(mad_(nu));
    stats.push_back(rms_());
    if (positive_()) {
         float gmu = gmean_();
         stats.push_back(gmu);
         stats.push_back(gstd_(gmu));
         stats.push_back(hmean_());
    }
    return stats;
}

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
    for (int i = 0; i < n; i += 2) {
		float u = random_();
		float v = random_();
		float r = sigma * sqrt(-2.0 * log(u));
		float s = mu + r * cos(2.0 * PI * v);
		x_[i] = s;
		if (i + 1 < n) {
			float t = mu + r * sin(2.0 * PI * v);
			x_[i + 1] = t;
		}
    }
}

void dataSet::set_truncated_normal_(int n, float lo, float hi, float mu, float sigma)
{
	x_.clear();
	x_.resize(n);
	if (mu < lo || mu > hi) return;
	if (sigma > 0.5 * (hi - lo)) return;
    int count = 0;
	while (count < n) {
		float u = random_();
		float v = random_();
		float r = sigma * sqrt(-2.0 * log(u));
		float s = mu + r * cos(2.0 * PI * v);
		if (s >= lo && s <= hi) {
			x_[count] = s;
			++count;
		}
		if (count < n) {
			float t = mu + r * sin(2.0 * PI * v);
			if (t >= lo && t <= hi) {
				x_[count] = t;
				++count;
			}
		}
    }	
}

void dataSet::print_dataSet_() const
{
    std::vector<float>::const_iterator cit;
    for (cit = x_.begin(); cit != x_.end(); cit++) std::cout << *cit << std::endl;
}

void dataSet::report_() const
{
    // processing
    float minimum = getMin();
    float maximum = getMax();
    float range = maximum - minimum;
    float mu = mean_();
    float v = var_(mu);
    float s = sqrt(v);
    float rtmnsq = rms_();
    bool flag = positive_();
    float gmu;
    float hmu;
    float gsigma;
    if (flag) {
         gmu = gmean_();
         gsigma = gstd_(gmu);
         hmu = hmean_();
    }
    float nu = median_(); 
    float mdndev = mad_(nu);
    // output
    std::cout << "size = " << x_.size() << '\n';
    std::cout << "minimum = " << minimum << '\n';
    // std::cout << "minimum = " << qsel(x_, 0) << '\n';
    std::cout << "maximum = " << maximum << '\n';
    // std::cout << "maximum = " << qsel(x_, x_.size() - 1) << '\n';
    std::cout << "range = " << range << '\n';
    std::cout << "arithmetic mean = " << mu << '\n';
    std::cout << "sample variance = " << v << '\n';
    std::cout << "sample standard deviation = " << s << '\n';
    std::cout << "root mean square = " << rtmnsq << '\n';
    if (flag) {
		std::cout << "geometric mean = " << gmu << '\n';
		std::cout << "geometric standard deviation = " << gsigma << '\n';
		std::cout << "harmonic mean = " << hmu << '\n';
	}
	std::cout << "median = " << nu << '\n';
	std::cout << "median absolute deviation = " << mdndev << '\n';
    std::cout << std::endl;
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

float dataSet::rms_() const
{
   float acc = 0.0;
   std::vector<float>::const_iterator cit;
   for (cit = x_.begin(); cit != x_.end(); ++cit)
      acc += *cit * *cit;
   return sqrt(acc / x_.size());
}

float dataSet::gmean_() const
{
	float acc = 0.0;
	std::vector<float>::const_iterator cit;
	for (cit = x_.begin(); cit != x_.end(); ++cit)
		acc += log(*cit);
	return exp(acc / x_.size());
}

float dataSet::gstd_(float gmu) const
{
	float acc = 0.0;
	std::vector<float>::const_iterator cit;
	for (cit = x_.begin(); cit != x_.end(); ++cit) {
		// float term = log(*cit / gmu);
		// acc += term * term;
		acc += pow(log(*cit / gmu), 2);
	}
	return exp(sqrt(acc / x_.size()));
}

float dataSet::hmean_() const
{
	float acc = 0.0;
	std::vector<float>::const_iterator cit;
	for (cit = x_.begin(); cit != x_.end(); ++cit)
		acc += 1.0 / *cit;
	return x_.size() / acc;
}

bool dataSet::positive_() const
{
   std::vector<float>::const_iterator cit;
   for (cit = x_.begin(); cit != x_.end(); ++cit)
      if (*cit <= 0)
         return false;
   return true;
}

float dataSet::median_() const
{
	int mid = x_.size() / 2;
	if (x_.size() % 2 == 1)
		return qsel(x_, mid);
	else 
		return 0.5 * (qsel(x_, mid - 1) + qsel(x_, mid));
}

float dataSet::mad_(float nu) const
{
	std::vector<float> dev(x_.size());
	std::vector<float>::const_iterator cit;
	int i = 0;
	for (cit = x_.begin(); cit != x_.end(); ++cit)
		dev[i++] = fabs(*cit - nu);
	int mid = dev.size() / 2;
	if (dev.size() % 2 == 1)
		return qsel(dev, mid);
	else 
		return 0.5 * (qsel(dev, mid - 1) + qsel(dev, mid));
}
