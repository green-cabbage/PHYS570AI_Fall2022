#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 30
#define N_LAYER_3 50
#define N_LAYER_6 50
#define N_LAYER_9 50
#define N_LAYER_12 150
#define N_LAYER_15 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<26,14> model_default_t;
typedef ap_fixed<26,14> input_t;
typedef ap_fixed<26,14> layer2_t;
typedef ap_fixed<26,14> layer3_t;
typedef ap_fixed<26,14> layer5_t;
typedef ap_fixed<26,14> layer6_t;
typedef ap_fixed<26,14> layer8_t;
typedef ap_fixed<26,14> layer9_t;
typedef ap_fixed<26,14> layer11_t;
typedef ap_fixed<26,14> layer12_t;
typedef ap_fixed<26,14> layer14_t;
typedef ap_fixed<26,14> layer15_t;
typedef ap_fixed<26,14> result_t;

#endif
