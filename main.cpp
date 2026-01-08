#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "network.hpp"
    
int main(int argc, char *argv[])
{
    std::string interpret_output(const std::vector<float>&, 
		const std::vector<std::string>&);
    void parse_CSV(const std::string&, std::vector<std::vector<float> >&, 
        std::vector<std::vector<float> >&, std::vector<std::string>&, 
        std::vector<std::string>&);
    void dump(int, const std::vector<int>&, float, int, int);

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
		return 1;
	}
  
    srand(time(0));

    std::vector<std::vector<float> > parameters;
    std::vector<std::vector<float> > classifications;
    std::vector<std::string> labels, headers;
    
    parse_CSV(argv[1], parameters, classifications, labels, headers);
    std::vector<int> layer_structure(3);

    // iris dataset training and testing
    normalize_by_feature_scaling(parameters);

    // trainers
    int n = 0.8 * parameters.size();
    std::vector<std::vector<float> > trainers(n);
    std::vector<std::vector<float> > trainers_corrects(n);
    for (int i = 0; i < n; i++) trainers[i] = parameters[i];
    for (int i = 0; i < n; i++) trainers_corrects[i] = classifications[i];

    // testers
    std::vector<std::vector<float> > testers(parameters.begin() + n, parameters.end());
    std::vector<std::string> testers_corrects(labels.begin() + n, labels.end());
    int siz = testers_corrects.size();
    
    /*
    int x = parameters[0].size();
    int z = headers.size();
    int y = 6 + ((2.0 / 3.0) * x + z) / 10.0;
    int m = std::max(double(y + 1), ceil(x / 2));
    layer_structure[0] = x;
    layer_structure[2] = z;
	std::vector<float> results((m - 1) * 9);
    */
    layer_structure[0] = parameters[0].size();
    layer_structure[2] = headers.size();
    int x = (2.0 / 3.0) * layer_structure[0] + layer_structure[2];
    int y = ceil(layer_structure[0] / 2.0) - 1;
    int lo = (x < y) ? x : y;
    int hi = (x < y) ? y : x;
    // std::cout << x << ' ' << y << ' ' << m << std::endl;
    std::vector<float> results(9 * (hi - lo + 1));
	int j = 0;
    for (int nn = lo; nn <= hi; nn++) {
        float lr = 0.9;
        for (int i = 0; i < 9; i++) {
            layer_structure[1] = nn;
            Network network(layer_structure, lr);
            for (int i = 0; i < 50; i++) network.Train(trainers, trainers_corrects);
	        int correct = network.Validate(testers, testers_corrects, headers, &interpret_output);
            dump(parameters.size(), layer_structure, lr, correct, siz);
            lr -= 0.1;
            results[j++] = correct / float(siz);
        }
    }
    dataSet stats(results);
    std::cout << "min = " << stats.getMin() << std::endl;
    std::cout << "max = " << stats.getMax() << std::endl;
    std::cout << "mean = " << stats.getMean() << std::endl;
    std::cout << "std = " << stats.getStd() << std::endl;
    return 0;
}

std::string interpret_output(const std::vector<float>& output, 
									const std::vector<std::string>& headers)
{
    std::vector<float>::const_iterator loc;
    loc = std::max_element(output.begin(), output.end());
    for (int i = 0; i < int(headers.size()); i++)
		if (*loc == output[i])
			return headers[i];
	return headers[0];
}


void parse_CSV(const std::string& filename, 
				std::vector<std::vector<float> >& parameters, 
				std::vector<std::vector<float> >& classifications, 
				std::vector<std::string>& labels,
				std::vector<std::string>& headers)
{
    std::vector<std::vector<std::string> > dataset;
    std::string line;
    std::ifstream fin(filename.c_str());
    if (!fin) {
        std::cerr << "error opening file" << std::endl;
        std::exit(2);
    }
    while (std::getline(fin, line)) {
        std::istringstream ss(line);
        std::vector<std::string> record;
        std::string field;
        while (std::getline(ss, field, ','))
            record.push_back(field);
        dataset.push_back(record);
    }
    fin.close();
    // shuffle records for random order
    std::random_shuffle(dataset.begin(), dataset.end());
    int m = dataset[0].size();
    int label_column = 0;
    for (int i = 0; i < m; i++)
		if (!isdigit(dataset[0][i][0]) && dataset[0][i][0] != '.') {
			label_column = i;
			break;
		}
    
    int n = dataset.size();
    parameters.resize(n);
    classifications.resize(n);
    labels.resize(n);
    for (int i = 0; i < n; i++) {
        std::vector<float> record_parameters(m - 1);
        int k = 0;
        for (int j = 0; j < m; j++) {
			if (j == label_column) {
				labels[i] = std::string(dataset[i][j]);
				if (std::find(headers.begin(), headers.end(), labels[i]) == headers.end())
					headers.push_back(labels[i]);
			}
			else {
				std::istringstream ss(dataset[i][j]);
				float f;
				ss >> f;
				record_parameters[k++] = f;
            }
        }
        parameters[i] = record_parameters;
    }
    for (int i = 0; i < n; i++) {
		classifications[i] = std::vector<float>(headers.size(), 0);
		std::vector<std::string>::const_iterator loc;
		loc = std::find(headers.begin(), headers.end(), labels[i]);
		int idx = loc - headers.begin();
		classifications[i][idx] = 1.0;
	}
}

void dump(int n, const std::vector<int>& ls, float lr, int corr, int siz)
{
    float pct = corr / float(siz);
    std::cout << n << ", ";
    std::vector<int>::const_iterator cit;
    for (cit = ls.begin(); cit != ls.end(); cit++)
        std::cout << *cit << ", ";
    std::cout << std::setprecision(1) << lr << ", ";
    std::cout << n - siz << ", ";
    std::cout << corr << ", ";
    std::cout << siz << ", ";
    std::cout << std::setprecision(3) << pct << std::endl;
}
