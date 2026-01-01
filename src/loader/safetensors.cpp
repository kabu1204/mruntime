#include "mruntime/safetensors.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "MLAS/include/mlas_float16.h"

namespace mruntime {

namespace {

DType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F16") return DType::FP16;
    if (dtype_str == "BF16") return DType::BF16;
    if (dtype_str == "F32") return DType::FP32;
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

}  // namespace

void SafeTensorsFile::parse_header(const char* header_data, size_t header_size) {
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_data, header_data + header_size);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse SafeTensors header JSON: ") + e.what());
    }

    if (!header.is_object()) {
        throw std::runtime_error("SafeTensors header JSON must be an object");
    }

    for (const auto& [name, tensor_obj] : header.items()) {
        if (name == "__metadata__") continue;

        try {
            if (!tensor_obj.is_object()) {
                throw std::runtime_error("tensor entry is not an object");
            }

            TensorInfo info;
            info.name = name;
            info.dtype = parse_dtype(tensor_obj.at("dtype").get<std::string>());
            info.shape = tensor_obj.at("shape").get<std::vector<size_t>>();

            std::vector<size_t> offsets = tensor_obj.at("data_offsets").get<std::vector<size_t>>();
            if (offsets.size() != 2) {
                throw std::runtime_error("data_offsets must have exactly 2 elements");
            }
            info.data_offset = offsets[0];
            info.data_size = offsets[1] - offsets[0];

            tensors_[name] = std::move(info);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse tensor info for '" + name + "': " + e.what());
        }
    }
}

std::unique_ptr<SafeTensorsFile> SafeTensorsFile::open(const std::string& path) {
    auto file = std::unique_ptr<SafeTensorsFile>(new SafeTensorsFile());
    file->path_ = path;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    file->file_size_ = static_cast<size_t>(fileSize.QuadPart);

    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
        CloseHandle(hFile);
        throw std::runtime_error("Failed to create file mapping: " + path);
    }
    file->mapped_data_ = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    CloseHandle(hFile);
    if (!file->mapped_data_) {
        throw std::runtime_error("Failed to map file: " + path);
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        ::close(fd);
        throw std::runtime_error("Failed to stat file: " + path);
    }
    file->file_size_ = static_cast<size_t>(st.st_size);

    file->mapped_data_ = mmap(nullptr, file->file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (file->mapped_data_ == MAP_FAILED) {
        file->mapped_data_ = nullptr;
        throw std::runtime_error("Failed to mmap file: " + path);
    }
#endif

    if (file->file_size_ < 8) {
        throw std::runtime_error("File too small to be SafeTensors: " + path);
    }

    const char* data = static_cast<const char*>(file->mapped_data_);
    uint64_t header_size;
    std::memcpy(&header_size, data, sizeof(header_size));
    if (header_size > file->file_size_ - 8) {
        throw std::runtime_error("Invalid header size in SafeTensors file: " + path);
    }

    file->data_offset_ = 8 + header_size;
    file->parse_header(data + 8, header_size);

    return file;
}

SafeTensorsFile::~SafeTensorsFile() {
    if (mapped_data_) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data_);
#else
        munmap(mapped_data_, file_size_);
#endif
    }
}

bool SafeTensorsFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorInfo& SafeTensorsFile::tensor_info(const std::string& name) const {
    return tensors_.at(name);
}

std::vector<std::string> SafeTensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

Tensor SafeTensorsFile::load_tensor_view(const std::string& name) const {
    const TensorInfo& info = tensor_info(name);
    void* data = static_cast<char*>(mapped_data_) + data_offset_ + info.data_offset;
    return Tensor::from_buffer(data, Shape(info.shape), info.dtype, false);
}

Tensor SafeTensorsFile::load_tensor_copy(const std::string& name) const {
    const TensorInfo& info = tensor_info(name);
    Tensor result = Tensor::empty(Shape(info.shape), info.dtype);
    const void* src = static_cast<const char*>(mapped_data_) + data_offset_ + info.data_offset;
    std::memcpy(result.data(), src, info.data_size);
    return result;
}

Tensor SafeTensorsFile::load_tensor_copy(const std::string& name, DType target_dtype) const {
    Tensor result = load_tensor_copy(name);
    if (target_dtype != result.dtype()) {
        result = result.to(target_dtype);
    }
    return result;
}

}  // namespace mruntime
