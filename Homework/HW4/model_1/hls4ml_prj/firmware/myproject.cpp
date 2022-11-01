//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t norm_input[N_INPUT_1_1],
    result_t layer17_out[N_LAYER_15],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=norm_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=norm_input,layer17_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_15;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 30>(s2, "s2.txt");
        nnet::load_weights_from_txt<model_default_t, 30>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 1500>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 50>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 2500>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 50>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 2500>(w9, "w9.txt");
        nnet::load_weights_from_txt<model_default_t, 50>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 7500>(w12, "w12.txt");
        nnet::load_weights_from_txt<model_default_t, 150>(b12, "b12.txt");
        nnet::load_weights_from_txt<model_default_t, 150>(w15, "w15.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b15, "b15.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(norm_input, layer2_out, s2, b2); // norm

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // input_layer

    layer5_t layer5_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out); // activation_0

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // fc0

    layer8_t layer8_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out); // activation_1

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9); // fc1

    layer11_t layer11_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer9_t, layer11_t, relu_config11>(layer9_out, layer11_out); // activation_2

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // fc2

    layer14_t layer14_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::relu<layer12_t, layer14_t, relu_config14>(layer12_out, layer14_out); // activation_output

    layer15_t layer15_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::dense<layer14_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // output

    nnet::sigmoid<layer15_t, result_t, sigmoid_config17>(layer15_out, layer17_out); // sigmoid

}
