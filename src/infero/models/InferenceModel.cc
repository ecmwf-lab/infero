/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <vector>
#include <string>

#include "eckit/exception/Exceptions.h"
#include "eckit/config/LocalConfiguration.h"
#include "eckit/filesystem/LocalPathName.h"
#include "eckit/mpi/Comm.h"


#include "infero/models/InferenceModel.h"


using namespace eckit;

namespace infero {

// Configuration and model-specific defaults
InferenceModel::InferenceModel(const eckit::Configuration& conf, const eckit::Configuration& defaults) :
    Configurable(conf.getSubConfiguration("model_config"), defaults),
    modelBuffer_{size_t(0)},
    modelType_{conf.getString("type")},
    modelPath_{conf.getString("path")},
    isOpen_{false} {
}

InferenceModel::~InferenceModel() {

    if(isOpen_){
        close();
    }

    print_statistics();
}

std::string InferenceModel::name() const
{
    return std::string();
}

void InferenceModel::open()  {

    // soft check: multiple open() allowed
    if (isOpen_){
        Log::info() << "INFO: Inference model already open.. " << std::endl;
    } else {
        isOpen_ = true;
    }
}

void InferenceModel::infer(linalg::TensorFloat& tIn, linalg::TensorFloat& tOut, const std::string& input_name, const std::string& output_name)
{

    std::lock_guard<std::mutex> lock(modelMutex_);

    // Input Tensor re-ordering as needed
    eckit::Timing t_start(statistics_.timer());
    eckit::linalg::TensorFloat input_tensor;

    if (tIn.layout()==eckit::linalg::TensorFloat::Layout::ColMajor) {
        input_tensor = tIn.transformColMajorToRowMajor();
    } else {

        // TODO: this still makes a copy (for now)
        input_tensor = tIn;
    }
    statistics_.iTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;

    // do the actual inference..
    eckit::Timing start_infer(statistics_.timer());

    if ( !input_name.empty() || !output_name.empty()){

        // input/output names provided
        infer_impl(input_tensor, tOut, input_name, output_name);

    } else {

        // use defaults
        infer_impl(input_tensor, tOut);
    }

    statistics_.inferenceTiming_ += eckit::Timing{statistics_.timer()} - start_infer;


}

void InferenceModel::infer_impl(linalg::TensorFloat& tIn, linalg::TensorFloat& tOut, std::string input_name, std::string output_name)
{
    NOTIMP;
}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{
    std::lock_guard<std::mutex> lock(modelMutex_);

    // Take copy of the input tensors
    std::vector<eckit::linalg::TensorFloat*> inputTensors(tIn.begin(), tIn.end());

    // For each tensor that needs re-ordering, do it into a copy
    std::vector<std::unique_ptr<eckit::linalg::TensorFloat>> temporaryCopies;

    eckit::Timing t_start(statistics_.timer());
    for (int i = 0; i < inputTensors.size(); ++i) {
        if (inputTensors[i]->layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {
            temporaryCopies.emplace_back(new eckit::linalg::TensorFloat(inputTensors[i]->transformColMajorToRowMajor()));
            inputTensors[i] = temporaryCopies.back().get();
        }
    }
    statistics_.iTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;

    // do the actual inference..
    eckit::Timing start_infer(statistics_.timer());
    infer_mimo_impl(inputTensors, input_names, tOut, output_names);
    statistics_.inferenceTiming_ += eckit::Timing{statistics_.timer()} - start_infer;

}


void InferenceModel::infer_mimo(const TensorMap& iMap, const TensorMap& oMap) {

    std::vector<eckit::linalg::TensorFloat*> input_tensors;
    std::vector<const char*> input_names;
    for(auto& p: iMap){
        input_names.push_back(p.first.c_str());
        input_tensors.push_back(p.second);        
    }

    std::vector<eckit::linalg::TensorFloat*> output_tensors;
    std::vector<const char*> output_names;
    for(auto& p: oMap){
        output_names.push_back(p.first.c_str());
        output_tensors.push_back(p.second);        
    }

    infer_mimo(input_tensors, input_names, output_tensors, output_names);

}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*>& tIn, std::vector<const char*>& input_names,
                                     std::vector<eckit::linalg::TensorFloat*>& tOut, std::vector<const char*>& output_names)
{
    NOTIMP;
}

void InferenceModel::close() {

    // soft check: multiple close() allowed
    if (!isOpen_){
        Log::info() << "INFO: Inference model already closed.. " << std::endl;
    } else {
        isOpen_ = false;
    }
}

void InferenceModel::broadcast_model(const std::string path) {
    modelBuffer_ = eckit::mpi::comm().broadcastFile(path, 0);
}


void InferenceModel::print_statistics()
{
    Log::info() << statistics() << std::endl;
}


void InferenceModel::print_config()
{
    Log::info() << std::endl;
    Log::info() << "**** Infero Model Configuration ****" << std::endl;
    Log::info() << "Model type: " << modelType_ << std::endl;
    Log::info() << "Model path: " << modelPath_ << std::endl;
    Log::info() << "Configuration: " << std::endl;
    Log::info() << config() << std::endl;
    Log::info() << "************************************" << std::endl;
    Log::info() << std::endl;
}


//-------------------------------------------------------------------------------------------------


InferenceModelFactory::InferenceModelFactory() {}

InferenceModelFactory::~InferenceModelFactory() {}

InferenceModelFactory& InferenceModelFactory::instance() {
    static InferenceModelFactory theinstance;
    return theinstance;
}

void InferenceModelFactory::enregister(const std::string& name,
                                       const InferenceModelBuilderBase& builder) {
    std::lock_guard<std::mutex> lock(mutex_);
    ASSERT(builders_.find(name) == builders_.end());
    builders_.emplace(std::make_pair(name, std::ref(builder)));
}

void InferenceModelFactory::deregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = builders_.find(name);
    ASSERT(it != builders_.end());
    builders_.erase(it);
}

InferenceModel* InferenceModelFactory::build(const std::string& name,
                                             const eckit::Configuration& config) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = builders_.find(name);
    if (it == builders_.end()) {
        std::ostringstream ss;
        ss << "Builder not found for backend " << name;
        throw SeriousBug(ss.str(), Here());
    }

    return it->second.get().make(config);
}

InferenceModelBuilderBase::InferenceModelBuilderBase(const std::string& name) :
    name_(name) {
    InferenceModelFactory::instance().enregister(name, *this);
}

InferenceModelBuilderBase::~InferenceModelBuilderBase() {
    InferenceModelFactory::instance().deregister(name_);
}






}  // namespace infero
