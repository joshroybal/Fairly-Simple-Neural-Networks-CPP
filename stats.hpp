#ifndef STATS_H
#define STATS_H

#include <vector>
#include <algorithm>
#include <cmath>

class dataSet {
    public:
        std::vector<float> getdataSet() const { return x_; };
        int getSize() const { return x_.size(); };
        float getMin() const 
        { return *std::min_element(x_.begin(), x_.end()); };
        float getMax() const 
        { return *std::max_element(x_.begin(), x_.end()); };
        float getMean() const { return mean_(); };
        float getStd() const { return sqrt(var_(mean_())); };
        void setUniform(int n) { set_uniform_(n); };
        void setNormal(int n, float m, float s) { set_normal_(n, m, s); };
        void setscaledNormal(int n) { set_scaled_normal_(n); };
        void printdataSet() const { print_dataSet_(); };
    private:
        std::vector<float> x_;
        float random_() const;
        void set_uniform_(int);
        void set_normal_(int, float, float);
        void set_scaled_normal_(int);
        void print_dataSet_() const;
        float mean_() const;
        float var_(float) const;
};

#endif
