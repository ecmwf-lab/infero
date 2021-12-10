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

#include "infero/models/InferenceModelONNX.h"
#include "infero/infero_utils.h"
#include "InferenceModelONNX.h"


using namespace eckit;

namespace infero {


InferenceModelONNX::InferenceModelONNX(const eckit::Configuration& conf) :
    InferenceModel(conf){

    std::string ModelPath(conf.getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    // environment
    env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model"));

    // Session options
    session_options = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions);
    session_options->SetIntraOpNumThreads(1);
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // if not null, use the model buffer
    if (modelBuffer_.size()){
        Log::info() << "Constructing ONNX model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(modelBuffer_.size()) << std::endl;
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env,
                                                                 modelBuffer_.data(),
                                                                 modelBuffer_.size(),
                                                                 *session_options));
    } else {  // otherwise construct from model path
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env, ModelPath.c_str(), *session_options));
    }


    // setup input/output interface
    setupInputLayers();
    setupOutputLayers();
}

InferenceModelONNX::~InferenceModelONNX() {

    for (auto& n: inputNames){
        free (n);
    }

    for (auto& n: outputNames){
        free (n);
    }
}

void InferenceModelONNX::infer(TensorFloat& tIn, TensorFloat& tOut,
                               std::string input_name, std::string output_name) {

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // only one input/output usable here
    ASSERT(inputNames.size() == 1);
    ASSERT(outputNames.size() == 1);

    if (tIn.isRight()) {
        Log::info() << "Input Tensor has right-layout, but left-layout is needed. "
                    << "Transforming to left.." << std::endl;
        tIn.toLeftLayout();
    }

    auto shape_64 = utils::convert_shape<size_t, int64_t>(tIn.shape());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                            tIn.data(),
                                            tIn.size(),
                                            shape_64.data(),
                                            shape_64.size());
    ASSERT(input_tensor.IsTensor());

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       inputNames.data(),
                                       &input_tensor,
                                       numInputs,
                                       outputNames.data(),
                                       numOutputs);

    // output tensors
    ASSERT(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    if (tOut.isRight()) {

         // ONNX uses Left (C) tensor layouts, so we need to convert
         auto out_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
         TensorFloat tLeft(output_tensors.front().GetTensorData<float>(),
                           utils::convert_shape<int64_t, size_t>(out_shape),
                           false);  // wrap data

         tOut = tLeft.transformLeftToRightLayout();
    }

    else {
         // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
         memcpy(tOut.data(), output_tensors.front().GetTensorData<float>(),
                output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
    }

}

void InferenceModelONNX::infer_mimo(std::vector<TensorFloat *> tIn, std::vector<const char *> input_names,
                                    std::vector<TensorFloat *> tOut, std::vector<const char *> output_names)
{

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Make a copy to keep input data in a consistent state
    std::vector<TensorFloat> itensors(tIn.size());
    for (int i=0; i<tIn.size(); i++){
        itensors[i] = *tIn[i];
    }

    // N Input tensors
    size_t NInputs = input_names.size();
    for (size_t i=0; i<NInputs; i++){

        if (itensors[i].isRight()) {
            Log::info() << i << "-th Input Tensor has right-layout, but left-layout is needed. "
                        << "Transforming to left.." << std::endl;
            itensors[i].toLeftLayout();
        }

        auto shape_64 = utils::convert_shape<size_t, int64_t>(itensors[i].shape());
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                itensors[i].data(),
                                                itensors[i].size(),
                                                shape_64.data(),
                                                shape_64.size());
        ASSERT(input_tensor.IsTensor());

        inputTensors.emplace_back(std::move(input_tensor));

    }

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       inputNames.data(),
                                       inputTensors.data(),
                                       numInputs,
                                       outputNames.data(),
                                       numOutputs);

    // output tensors
    ASSERT(output_tensors.size() == numOutputs);

    for (size_t i=0; i<numOutputs; i++){

         ASSERT(output_tensors[i].IsTensor());

         if (tOut[i]->isRight()) {

             // ONNX uses Left (C) tensor layouts, so we need to convert
             auto out_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
             TensorFloat tLeft(output_tensors[i].GetTensorData<float>(),
                               utils::convert_shape<int64_t, size_t>(out_shape),
                               false);  // wrap data

             *tOut[i] = tLeft.transformLeftToRightLayout();
         }
         else {
             // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
             memcpy(tOut[i]->data(), output_tensors[i].GetTensorData<float>(),
                    output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
         }

    }

}


void InferenceModelONNX::setupInputLayers() {

    // get input name
    numInputs = session->GetInputCount();

    for (size_t i=0; i<numInputs; i++){

        char* inputName_ = session->GetInputName(i, allocator);
        inputNames.push_back(inputName_);

        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> inputLayerShape_ = tensor_info.GetShape();
        inputLayerShapes.push_back(inputLayerShape_);
    }
}


void InferenceModelONNX::setupOutputLayers() {

    // get input name
    numOutputs = session->GetOutputCount();

    for (size_t i=0; i<numOutputs; i++){

        char* outputName_ = session->GetOutputName(i, allocator);
        outputNames.push_back(outputName_);

        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> outputLayerShape_ = tensor_info.GetShape();
        outputLayerShapes.push_back(outputLayerShape_);

    }
}


void InferenceModelONNX::print(std::ostream& os) const {

    os << "ONNX model has: " << numInputs << " inputs" << std::endl;
    for (size_t i=0; i<numInputs; i++){
        Log::info() << "Layer [" << i << "] " << inputNames[i] << " has shape: ";
        for (auto s: inputLayerShapes[i]){
            Log::info() << s << ", ";
        }
        Log::info() << std::endl;
    }

    os << "ONNX model has: " << numOutputs << " outputs" << std::endl;
    for (size_t i=0; i<numOutputs; i++){
        Log::info() << "Layer [" << i << "] " << outputNames[i] << " has shape: ";
        for (auto s: outputLayerShapes[i]){
            Log::info() << s << ", ";
        }
        Log::info() << std::endl;
    }

}


void InferenceModelONNX::print_shape(const Ort::Value& t){
    Ort::TensorTypeAndShapeInfo info = t.GetTensorTypeAndShapeInfo();
    for (auto i : info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;
}

}  // namespace infero
