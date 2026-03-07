#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <json.hpp>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

struct STTensor
{
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t offset0;
    int64_t offset1;
};

struct SafeTensors
{
    std::unordered_map<std::string, STTensor> tensors;
    void *data = nullptr;
    size_t mmapSize = 0;
    int fd = -1;

    ~SafeTensors()
    {
        if (data)
            munmap(data, mmapSize);

        if (fd >= 0)
            close(fd);
    }
};

inline SafeTensors load_safetensors(const std::string &path)
{
    using json = nlohmann::json;

    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open failed");

    uint64_t header_len = 0;
    f.read(reinterpret_cast<char *>(&header_len), 8);

    if (header_len == 0 || header_len > (1ull << 30))
        throw std::runtime_error("bad header size");

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    SafeTensors st;

    auto j = nlohmann::json::parse(header);

    for (auto &[name, v] : j.items())
    {
        bool valid =
            v.is_object() &&
            v.contains("dtype") &&
            v.contains("shape") &&
            v.contains("data_offsets");

        if (valid)
        {
            STTensor t;
            t.dtype = v["dtype"].get<std::string>();
            t.shape = v["shape"].get<std::vector<int64_t>>();

            auto off = v["data_offsets"];
            t.offset0 = off[0];
            t.offset1 = off[1];

            st.tensors[name] = t;
        }
    }

    // Open fd for mmap
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("open failed");

    // Get file size
    f.seekg(0, std::ios::end);
    size_t fileSize = size_t(f.tellg());

    size_t dataOffset = 8 + header_len;
    size_t dataSize = fileSize - dataOffset;

    size_t pageSize = sysconf(_SC_PAGE_SIZE);
    size_t alignedOffset = dataOffset & ~(pageSize - 1);
    size_t delta = dataOffset - alignedOffset;

    void *mapped = mmap(
        nullptr,
        dataSize + delta,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        alignedOffset);

    if (mapped == MAP_FAILED)
        throw std::runtime_error("mmap failed");

    st.data = reinterpret_cast<char *>(mapped) + delta;
    st.mmapSize = dataSize + delta;
    st.fd = fd;

    return st;
}