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
                   std::vector<std::vector<float> >& expecteds)
            { return train_(inputs, expecteds); };
        int Validate(std::vector<std::vector<float> >& inputs,
                     std::vector<std::string>& expecteds,
                     const std::vector<std::string>& headers,
                     std::string (*interpret_output)(const std::vector<float>& output,
                     const std::vector<std::string>& headers))
            { return validate_(inputs, expecteds, headers, interpret_output); };
        void Dump() { dump_(); };
    private:
        void init_(const std::vector<int>& layer_structure, float learning_rate);
        float learning_rate_;
        std::vector<Layer> layers_;
        Layer input_layer_;
        std::vector<float> outputs_(const std::vector<float>&);
        void backpropagate_(const std::vector<float>&);
        void update_weights_();
        void dump_();
        void train_(std::vector<std::vector<float> >&,
                    std::vector<std::vector<float> >&);
        int validate_(std::vector<std::vector<float> >&,
                     std::vector<std::string>&,
                     const std::vector<std::string>&,
                     std::string (*)(const std::vector<float>&, 
                     const std::vector<std::string>&));
};

#endif
