// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/c/openvino.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        char* err_msg = (char*)ov_get_last_err_msg();                                    \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        if (err_msg) {                                                                   \
            fprintf(stderr, "[ERROR] OpenVINO error: %s\n", err_msg);                    \
            ov_free(err_msg);                                                            \
        }                                                                                \
        goto err;                                                                        \
    }

int main(int argc, char** argv) {
    // -------- Check input parameters --------
    if (argc != 3) {
        printf("Usage : ./gemma_mul_sample <path_to_model> <device_name>\n");
        return EXIT_FAILURE;
    }

    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_tensor_t* input_tensor1 = NULL;
    ov_tensor_t* input_tensor2 = NULL;
    ov_tensor_t* output_tensor = NULL;
    ov_shape_t input_shape = {0};
    
    // -------- Parsing and validation of input arguments --------
    const char* model_path = argv[1];
    const char* device_name = argv[2];

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    CHECK_STATUS(ov_core_create(&core));

    // -------- Step 2. Read a model --------
    printf("[INFO] Loading model file: %s\n", model_path);
    CHECK_STATUS(ov_core_read_model(core, model_path, NULL, &model));

    // -------- Step 3. Set up input shapes and tensors --------
    int64_t dims[] = {2, 2};
    CHECK_STATUS(ov_shape_create(2, dims, &input_shape));

    // Create first input tensor
    CHECK_STATUS(ov_tensor_create(F32, input_shape, &input_tensor1));
    float* data1 = NULL;
    CHECK_STATUS(ov_tensor_data(input_tensor1, (void**)&data1));
    data1[0] = 2.0f; data1[1] = 2.0f; data1[2] = 3.0f; data1[3] = 4.0f;

    // Create second input tensor
    CHECK_STATUS(ov_tensor_create(F32, input_shape, &input_tensor2));
    float* data2 = NULL;
    CHECK_STATUS(ov_tensor_data(input_tensor2, (void**)&data2));
    data2[0] = 5.0f; data2[1] = 6.0f; data2[2] = 7.0f; data2[3] = 8.0f;

    // -------- Step 4. Loading a model to the device --------
    CHECK_STATUS(ov_core_compile_model(core, model, device_name, 0, &compiled_model));

    // -------- Step 5. Create an infer request --------
    CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

    // -------- Step 6. Prepare input --------
    // Assuming inputs are at index 0 and 1
    CHECK_STATUS(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor1));
    CHECK_STATUS(ov_infer_request_set_input_tensor_by_index(infer_request, 1, input_tensor2));

    // -------- Step 7. Do inference synchronously --------
    printf("[INFO] Starting inference...\n");
    CHECK_STATUS(ov_infer_request_infer(infer_request));

    // -------- Step 8. Process output --------
    CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    
    float* output_data = NULL;
    CHECK_STATUS(ov_tensor_data(output_tensor, (void**)&output_data));
    
    printf("[INFO] Inference results:\n");
    for (size_t i = 0; i < 4; ++i) {
        printf("Result[%zu]: %.2f\n", i, output_data[i]);
    }

err:
    if (output_tensor) ov_tensor_free(output_tensor);
    if (input_tensor1) ov_tensor_free(input_tensor1);
    if (input_tensor2) ov_tensor_free(input_tensor2);
    ov_shape_free(&input_shape);
    if (infer_request) ov_infer_request_free(infer_request);
    if (compiled_model) ov_compiled_model_free(compiled_model);
    if (model) ov_model_free(model);
    if (core) ov_core_free(core);
    
    return EXIT_SUCCESS;
}
