#pragma once

#include <memory>
#include <sstream>

#include "eckit/serialisation/Stream.h"

#include "eckit/linalg/Tensor.h"

using namespace eckit::linalg;

namespace infero {

class MLTensor : public TensorFloat {

public:
    // useful to convert shape of size_t to/from sahpe of int64_t
    template <typename F, typename T>
    static std::vector<T> convert_shape(const std::vector<F>& vec) {
        std::vector<T> vec_new;
        std::copy(vec.begin(), vec.end(), back_inserter(vec_new));
        return vec_new;
    }

    // error type for tensor comparison
    enum ErrorType
    {
        MSE
    };

public:

    MLTensor(bool isRight = true);

    // takes no ownership of memory
    MLTensor(const float* array, const std::vector<Size>& shape, bool isRight = true);

    // has ownership of memory
    MLTensor(const std::vector<Size>& shape, bool isRight = true);

    // shape as std::vector<size_t>
    std::vector<Size> shape() const { return shape_; }

    // from/to file
    static std::unique_ptr<MLTensor> from_file(const std::string& filename);
    void to_file(const std::string& filename);

    // compare against another tensor
    float compare(MLTensor& other, ErrorType mes = MSE) const;

private:
    // from/to formatted CSV
    static std::unique_ptr<MLTensor> from_csv(const std::string& filename);
    void to_csv(const std::string& filename);

    // from numpy saved as npy
    static std::unique_ptr<MLTensor> from_numpy(const std::string& filename);
    void to_numpy(const std::string& filename);

};

}  // namespace infero
