#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/w15.h"
#include "weights/b15.h"

//hls-fpga-machine-learning insert layer-config
// norm
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// input_layer
struct config3 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_3;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 1500;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<26,14> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_0
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// fc0
struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_3;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 2500;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<26,14> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_1
struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// fc1
struct config9 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_9;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 2500;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<26,14> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_2
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// fc2
struct config12 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_9;
    static const unsigned n_out = N_LAYER_12;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 7500;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<26,14> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_output
struct relu_config14 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// output
struct config15 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned n_out = N_LAYER_15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 150;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<26,14> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// sigmoid
struct sigmoid_config17 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
