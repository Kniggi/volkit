#include <algorithm>
#include <cassert>
#include <cstring> // memcpy

#include <vkt/linalg.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/StructuredVolume.h>

#include "StructuredVolume_impl.hpp"

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{
    StructuredVolume::StructuredVolume()
        : dims_(0, 0 , 0)
        , bytesPerVoxel_(1)
        , voxelMapping_(0.f, 1.f)
    {
    }

    StructuredVolume::StructuredVolume(
            int32_t dimx,
            int32_t dimy,
            int32_t dimz,
            uint16_t bytesPerVoxel,
            float mappingLo,
            float mappingHi
            )
        : dims_(dimx, dimy, dimz)
        , bytesPerVoxel_(bytesPerVoxel)
        , voxelMapping_(mappingLo, mappingHi)
    {
        ManagedBuffer::allocate(getSizeInBytes());
    }

    StructuredVolume::StructuredVolume(StructuredVolume& rhs)
        : dims_(rhs.dims_)
        , bytesPerVoxel_(rhs.bytesPerVoxel_)
        , voxelMapping_(rhs.voxelMapping_)
    {
        ManagedBuffer::allocate(getSizeInBytes());

        ManagedBuffer::copy((ManagedBuffer&)rhs);
    }

    StructuredVolume::~StructuredVolume()
    {
        ManagedBuffer::deallocate();
    }

    StructuredVolume& StructuredVolume::operator=(StructuredVolume& rhs)
    {
        if (&rhs != this)
        {
            dims_ = rhs.dims_;
            bytesPerVoxel_ = rhs.bytesPerVoxel_;
            voxelMapping_ = rhs.voxelMapping_;

            ManagedBuffer::deallocate();

            ManagedBuffer::allocate(getSizeInBytes());

            ManagedBuffer::copy((ManagedBuffer&)rhs);
        }

        return *this;
    }

    void StructuredVolume::setDims(int32_t dimx, int32_t dimy, int32_t dimz)
    {
        setDims(vec3i(dimx, dimy, dimz));
    }

    void StructuredVolume::setDims(vec3i dims)
    {
        dims_ = dims;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    vec3i StructuredVolume::getDims() const
    {
        return dims_;
    }

    void StructuredVolume::setBytesPerVoxel(uint16_t bpv)
    {
        bytesPerVoxel_ = bpv;
        std::size_t newSize = getSizeInBytes();

        resize(newSize);
    }

    uint16_t StructuredVolume::getBytesPerVoxel() const
    {
        return bytesPerVoxel_;
    }

    void StructuredVolume::setVoxelMapping(float lo, float hi)
    {
        voxelMapping_ = vec2f(lo, hi);
    }

    void StructuredVolume::setVoxelMapping(vec2f mapping)
    {
        voxelMapping_ = mapping;
    }

    vec2f StructuredVolume::getVoxelMapping() const
    {
        return voxelMapping_;
    }

    uint8_t* StructuredVolume::getData()
    {
        migrate();

        return ManagedBuffer::data_;
    }

    void StructuredVolume::setValue(int32_t x, int32_t y, int32_t z, float value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        mapVoxel(ManagedBuffer::data_ + index, value);
    }

    void StructuredVolume::getValue(int32_t x, int32_t y, int32_t z, float& value)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        unmapVoxel(value, ManagedBuffer::data_ + index);
    }

    void StructuredVolume::setVoxel(int32_t x, int32_t y, int32_t z, uint8_t const* data)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            ManagedBuffer::data_[index + i] = data[i];
    }

    void StructuredVolume::getVoxel(int32_t x, int32_t y, int32_t z, uint8_t* data)
    {
        migrate();

        size_t index = linearIndex(x, y, z);

        for (uint16_t i = 0; i < bytesPerVoxel_; ++i)
            data[i] = ManagedBuffer::data_[index + i];
    }

    void StructuredVolume::mapVoxel(uint8_t* dst, float src) const
    {
        src -= voxelMapping_.x;
        src /= voxelMapping_.y - voxelMapping_.x;

        switch (bytesPerVoxel_)
        {
            case 1:
            {
                uint8_t ival = src * 255.999f;
                dst[0] = ival;
                break;
            }

            case 2:
            {
                uint16_t ival = src * 65535.999f;
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
#else
                dst[0] = static_cast<uint8_t>(ival >> 8);
                dst[1] = static_cast<uint8_t>(ival);
#endif
            }

            case 4:
            {
                uint32_t ival = src * 4294967295.999f;
#ifdef VKT_LITTLE_ENDIAN
                dst[0] = static_cast<uint8_t>(ival);
                dst[1] = static_cast<uint8_t>(ival >> 8);
                dst[2] = static_cast<uint8_t>(ival >> 16);
                dst[3] = static_cast<uint8_t>(ival >> 24);
#else
                dst[0] = static_cast<uint8_t>(ival >> 24);
                dst[1] = static_cast<uint8_t>(ival >> 16);
                dst[1] = static_cast<uint8_t>(ival >> 8);
                dst[3] = static_cast<uint8_t>(ival);
#endif
            }
        }
    }

    void StructuredVolume::unmapVoxel(float& dst, uint8_t const* src) const
    {
        switch (bytesPerVoxel_)
        {
            case 1:
            {
                uint8_t ival = src[0];
                float fval = static_cast<float>(ival);
                dst = lerp(voxelMapping_.x, voxelMapping_.y, fval / 255.999f);
                break;
            }

            case 2:
            {
#ifdef VKT_LITTLE_ENDIAN
                uint16_t ival = static_cast<uint16_t>(src[0])
                              | static_cast<uint16_t>(src[1] << 8);
#else
                uint16_t ival = static_cast<uint16_t>(src[0] << 8)
                              | static_cast<uint16_t>(src[1]);
#endif
                float fval = static_cast<float>(ival);
                dst = lerp(voxelMapping_.x, voxelMapping_.y, fval / 65535.999f);
            }

            case 4:
            {
#ifdef VKT_LITTLE_ENDIAN
                uint32_t ival = static_cast<uint32_t>(src[0])
                              | static_cast<uint32_t>(src[1] << 8)
                              | static_cast<uint32_t>(src[2] << 16)
                              | static_cast<uint32_t>(src[3] << 24);
#else
                uint32_t ival = static_cast<uint32_t>(src[0] << 24)
                              | static_cast<uint32_t>(src[1] << 16)
                              | static_cast<uint32_t>(src[2] << 8)
                              | static_cast<uint32_t>(src[3]);
#endif
                float fval = static_cast<float>(ival);
                dst = lerp(voxelMapping_.x, voxelMapping_.y, fval / 4294967295.999f);
            }
        }
    }

    std::size_t StructuredVolume::getSizeInBytes() const
    {
        return dims_.x * std::size_t(dims_.y) * dims_.z * bytesPerVoxel_;
    }


    //--- private -----------------------------------------

    std::size_t StructuredVolume::linearIndex(int32_t x, int32_t y, int32_t z) const
    {
        size_t index = z * dims_.x * std::size_t(dims_.y)
                     + y * dims_.x
                     + x;
        return index * bytesPerVoxel_;
    }

} // vkt


//-------------------------------------------------------------------------------------------------
// C API
//

uint16_t vktStructuredVolumeGetMaxBytesPerVoxel()
{
    return vkt::StructuredVolume::GetMaxBytesPerVoxel();
}

void vktStructuredVolumeCreate(
        vktStructuredVolume* volume,
        int32_t dimx,
        int32_t dimy,
        int32_t dimz,
        uint16_t bytesPerVoxel,
        float mappingLo,
        float mappingHi
        )
{
    assert(volume != nullptr);

    *volume = new vktStructuredVolume_impl(dimx, dimy, dimz, bytesPerVoxel, mappingLo, mappingHi);
}

void vktStructuredVolumeCreateCopy(vktStructuredVolume* volume, vktStructuredVolume rhs)
{
    assert(volume != nullptr);

    *volume = new vktStructuredVolume_impl(rhs->volume);
}

void vktStructuredVolumeDestroy(vktStructuredVolume volume)
{
    delete volume;
}
