#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "network.hpp"
#include "util.hpp"
    
int main(int argc, char *argv[])
{
    void model(char *, float (*)(float), float (*)(float), 
            void (*)(std::vector<std::vector<float> >&));
    
    if (argc < 2) {
	    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
		return 1;
	}
  
    srand(time(0));
    model(argv[1], &sigmoid, &derivative_sigmoid, &normalize_by_z_score);
    float (*fnptr)(float) = &std::tanh;
    model(argv[1], fnptr, &derivative_tanh, &normalize_by_z_score);
    model(argv[1], &ReLU, &derivative_ReLU, &normalize_by_feature_scaling);
    model(argv[1], &leaky_ReLU, &derivative_leaky_ReLU, &normalize_by_feature_scaling);
    return 0;
}

void model(char* filename, float (*activation_function)(float),
        float (*derivative_activation_function)(float),
        void (*scaling_function)(std::vector<std::vector<float> >&))
{
   std::string interpret_output(const std::vector<float>&, 
	const std::vector<std::string>&);
   void parse_CSV(const std::string&, std::vector<std::vector<float> >&, 
      std::vector<std::vector<float> >&, std::vector<std::string>&, 
      std::vector<std::string>&);
   void dump(int, const std::vector<int>&, float, int, int);

    std::vector<std::vector<float> > parameters;
    std::vector<std::vector<float> > classifications;
    std::vector<std::string> labels, headers;

    parse_CSV(filename, parameters, classifications, labels, headers);
    std::vector<int> layer_structure(3);

    // iris dataset training and testing
    // normalize_by_feature_scaling(parameters);
    // normalize_by_z_scoring(parameters);
    scaling_function(parameters);

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
    

    int x = parameters[0].size();
    int z = headers.size();
    layer_structure[0] = x;
    layer_structure[2] = z;
    int m = std::max(12, int(std::ceil(x / 2)) - 1);
	 std::vector<float> results(m * 9);
     std::vector<std::vector<float> > table(m);
	 int j = 0;
    for (int nn = 1; nn <= m; nn++) {
        float lr = 0.9;
        std::vector<float> row(9);
        for (int i = 0; i < 9; i++) {
            layer_structure[1] = nn;
            Network network(layer_structure, lr);
            for (int i = 0; i < 10; i++) {
                network.Train(trainers, trainers_corrects, activation_function, 
                        derivative_activation_function);
            }
	        int correct = network.Validate(testers, testers_corrects, headers, 
                    &interpret_output, activation_function);
            float result = correct / float(siz);
            row[i] = result;
            results[j++] = result;
            lr -= 0.1;
        }
        // print_vector(row);
        table[nn-1] = row;
    }
    dataSet stats(results);
    print_vector(stats.getStats());
    /*
    std::vector<std::vector<float> >::const_iterator cit;
    for (cit = table.begin(); cit != table.end(); ++cit) {
         dataSet x(*cit);
         print_vector(x.getStats());
    }
    */
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
