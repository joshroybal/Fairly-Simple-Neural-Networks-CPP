#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "layer.hpp"

class Network {
    public:
        Network(const std::vector<int>& layer_structure, float learning_rate) :
            learning_rate_(learning_rate),
        input_layer_(0, layer_structure[0], learning_rate)
        { init_(layer_structure, learning_rate); };
        float getLearningRate() { return learning_rate_; };
        void Train(std::vector<std::vector<float> >& inputs,
                   std::vector<std::vector<float> >& expecteds,
                   float (*activation_function)(float),
                   float (*derivative_activation_function)(float))
            { return train_(inputs, expecteds, activation_function, derivative_activation_function); };
        int Validate(std::vector<std::vector<float> >& inputs,
                     std::vector<std::string>& expecteds,
                     const std::vector<std::string>& headers,
                     std::string (*interpret_output)(const std::vector<float>& output,
                     const std::vector<std::string>& headers),
                     float (*activation_function)(float))
            { return validate_(inputs, expecteds, headers, interpret_output, activation_function); };
        void Dump() { dump_(); };
    private:
        void init_(const std::vector<int>& layer_structure, float learning_rate);
        float learning_rate_;
        std::vector<Layer> layers_;
        Layer input_layer_;
        std::vector<float> outputs_(const std::vector<float>&, float (*)(float));
        void backpropagate_(const std::vector<float>&, float (*)(float));
        void update_weights_();
        void dump_();
        void train_(std::vector<std::vector<float> >&,
                std::vector<std::vector<float> >&, float (*)(float), 
                float (*)(float));
        int validate_(std::vector<std::vector<float> >&,
                     std::vector<std::string>&,
                     const std::vector<std::string>&,
                     std::string (*)(const std::vector<float>&, 
                     const std::vector<std::string>&), float (*)(float));
};

#endif
