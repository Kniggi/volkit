// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include "common.hpp"
#include "linalg.hpp"

namespace vkt
{
    struct VolumeFileHeader
    {
        bool isStructured = true;
        Vec3i dims = { 0, 0, 0 };
        uint16_t bytesPerVoxel = 0;
        Vec3f dist = { 1.f, 1.f, 1.f };
        Vec2f voxelMapping = { 0.f, 1.f };
    };

    enum class VolumeFileOpenMode
    {
        Read,
        Write,
        ReadWrite,
    };

    class VolumeFile : public DataSource
    {
    public:

        VolumeFile(char const* fileName, OpenMode om = OpenMode::Read);
       ~VolumeFile();

        std::size_t read(char* buf, std::size_t len);
        std::size_t write(char const* buf, std::size_t len);
        bool seek(std::size_t pos);
        bool flush();
        bool good() const;

        void setHeader(VolumeFileHeader header);
        VolumeFileHeader getHeader() const;

    private:
        VolumeFileHeader header_;

        DataSource* dataSource_ = nullptr;

        char const* fileName_ = nullptr;

    };

} // vkt
