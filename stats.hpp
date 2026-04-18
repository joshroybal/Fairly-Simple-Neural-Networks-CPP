#ifndef STATS_H
#define STATS_H

#include <vector>
#include <algorithm>
#include <cmath>

// functions and subroutines
float qsel(const std::vector<float>&, int);
int rndint(int, int);
int part(const std::vector<float>&, std::vector<int>&, int, int);
int rndint(int, int);
void swap(int&, int&);

class dataSet {
    public:
		dataSet() {};
		dataSet(const std::vector<float>& x) { x_ = x; };
        std::vector<float> getdataSet() const { return x_; };
        std::vector<float> getZScores() const { return z_scores_(); };
        std::vector<float> getStats() const { return get_stats_(); };
        int getSize() const { return x_.size(); };
        float getMin() const
        { return *std::min_element(x_.begin(), x_.end()); };
        float getMax() const
        { return *std::max_element(x_.begin(), x_.end()); };
        float getMean() const { return mean_(); };
        float getStd() const { return sqrt(var_(mean_())); };
        void setUniform(int n) { set_uniform_(n); };
        void setNormal(int n, float m=0.0, float s=1.0) { set_normal_(n, m, s); };
        void setTruncatedNormal(int n, float lo, float hi, float mu, float sigma)
        { return set_truncated_normal_(n, lo, hi, mu, sigma); };
        void printdataSet() const { print_dataSet_(); };
        void Report() const { report_(); };
    private:
        std::vector<float> x_;
        std::vector<float> z_scores_() const;
        std::vector<float> get_stats_() const;
        float random_() const;
        void set_uniform_(int);
        void set_normal_(int, float, float);
        void set_truncated_normal_(int, float, float, float, float);
        void print_dataSet_() const;
        void report_() const;
        float mean_() const;
        float var_(float) const;
        float rms_() const;
        float gmean_() const;
        float gstd_(float) const;
        float hmean_() const;
        bool positive_() const;
        float median_() const;
        float mad_(float) const;
};

#endif
