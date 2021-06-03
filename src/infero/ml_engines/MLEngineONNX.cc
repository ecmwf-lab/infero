/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "infero/ml_engines/MLEngineONNX.h"


using namespace eckit;

namespace infero {


MLEngineONNX::MLEngineONNX(std::string model_filename) : MLEngine(model_filename) {

    // environment
    env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model"));

    // Session options
    session_options = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions);
    session_options->SetIntraOpNumThreads(1);
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = std::unique_ptr<Ort::Session>(new Ort::Session(*env, mModelFilename.c_str(), *session_options));

    // query input/output layers
    query_input_layer();
    query_output_layer();
}

MLEngineONNX::~MLEngineONNX() {}

std::unique_ptr<infero::MLTensor> MLEngineONNX::infer(std::unique_ptr<infero::MLTensor>& input_sample) {

    // make a copy of the input data
    data_buffer.resize(input_sample->size());
    for (int i = 0; i < input_sample->size(); i++) {
        data_buffer[i] = input_sample->data()[i];
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto shape_64           = infero::MLTensor::convert_shape<size_t, int64_t>(input_sample->shape());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data_buffer.data(), input_sample->size(),
                                                              shape_64.data(), shape_64.size());
    ASSERT(input_tensor.IsTensor());

    Ort::TensorTypeAndShapeInfo info = input_tensor.GetTensorTypeAndShapeInfo();
    Log::info() << "Sample tensor shape: ";
    for (auto i : info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor,
                                       num_input_nodes, output_node_names.data(), num_output_nodes);

    ASSERT(output_tensors.size() == 1 && output_tensors.front().IsTensor());


    auto out_tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    int out_size         = 1;
    for (auto i : out_tensor_info.GetShape()) {
        out_size *= i;
    }

    Log::info() << "Prediction tensor shape: ";
    for (auto i : out_tensor_info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    auto shape_   = infero::MLTensor::convert_shape<int64_t, size_t>(out_tensor_info.GetShape());
    auto pred_ptr = std::unique_ptr<infero::MLTensor>(new infero::MLTensor(shape_, false));

    // copy output data
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_data(out_size);
    for (size_t i = 0; i < out_size; i++) {
        *(pred_ptr->data() + i) = *(floatarr + i);
    }

    return pred_ptr;
}


void MLEngineONNX::query_input_layer() {
    num_input_nodes = session->GetInputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_input_nodes == 1);
    input_node_idx = 0;

    // get input name
    input_name = session->GetInputName(input_node_idx, allocator);
    input_node_names.push_back(input_name);

    Ort::TypeInfo type_info                               = session->GetInputTypeInfo(input_node_idx);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    input_layer_shape = tensor_info.GetShape();
}


void MLEngineONNX::query_output_layer() {
    num_output_nodes = session->GetOutputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_output_nodes == 1);
    output_node_idx = 0;

    // print output node names
    output_name = session->GetOutputName(output_node_idx, allocator);
    output_node_names.push_back(output_name);

    // NOTE: this is the shape of the output tensor as described by the model
    // so it can be "dynamic" (with -1, meaning that accepts any tensor size on that axis)
    Ort::TypeInfo type_info                               = session->GetOutputTypeInfo(output_node_idx);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // print output shapes/dims
    output_layer_shape = tensor_info.GetShape();
}


void MLEngineONNX::print(std::ostream& os) const {

    os << "N input tensors: " << num_input_nodes << std::endl;
    os << "Input layer " << input_node_names[0] << " expects a Tensor with " << input_layer_shape.size()
       << " dimensions" << std::endl;

    for (int j = 0; j < input_layer_shape.size(); j++)
        os << "dim [" << j << "]: " << input_layer_shape[j] << std::endl;

    os << "N output tensors: " << num_output_nodes << std::endl;
    os << "Output layer " << output_node_names[0] << " expects a Tensor with " << output_layer_shape.size()
       << " dimensions" << std::endl;

    for (int j = 0; j < output_layer_shape.size(); j++)
        os << "dim [" << j << "]: " << output_layer_shape[j] << std::endl;
}

}  // namespace infero
