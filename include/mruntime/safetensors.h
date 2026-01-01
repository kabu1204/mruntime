#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mruntime/tensor.h"

namespace mruntime {

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<size_t> shape;
    size_t data_offset;
    size_t data_size;
};

class SafeTensorsFile {
public:
    static std::unique_ptr<SafeTensorsFile> open(const std::string& path);

    ~SafeTensorsFile();

    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    bool has_tensor(const std::string& name) const;
    const TensorInfo& tensor_info(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    // Zero-copy view into mmap'd file (read-only, valid while file is open)
    Tensor load_tensor_view(const std::string& name) const;

    // Owning copy with optional dtype conversion
    Tensor load_tensor_copy(const std::string& name, DType target_dtype) const;
    Tensor load_tensor_copy(const std::string& name) const;  // Keep original dtype

private:
    SafeTensorsFile() = default;
    void parse_header(const char* header_data, size_t header_size);

    std::map<std::string, TensorInfo> tensors_;
    std::string path_;
    void* mapped_data_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
};

}  // namespace mruntime
