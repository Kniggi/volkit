// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#if VKT_HAVE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif

#include <GL/glew.h>
#include <visionaray/math/simd/simd.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/io.h>
#include <visionaray/math/ray.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/scheduler.h>
#include <visionaray/swizzle.h>
#include <visionaray/thin_lens_camera.h>

#if VKT_HAVE_CUDA
#include <thrust/host_vector.h>
#include <visionaray/gpu_buffer_rt.h>
#endif

// Private visionaray_common includes!
#include <common/config.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/image.h>

#if VSNRAY_COMMON_HAVE_SDL2
#include <common/viewer_sdl2.h>
#else
#include <common/viewer_glut.h>
#endif

#include <vkt/ExecutionPolicy.hpp>
#include <vkt/HierarchicalVolume.hpp>
#include <vkt/LookupTable.hpp>
#include <vkt/Render.hpp>
#include <vkt/StructuredVolume.hpp>

#include <vkt/Render.h>
#include <vkt/StructuredVolume.h>

#include "HierarchicalVolumeView.hpp"
#include "Logging.hpp"
#include "Render_kernel.hpp"
#include "StructuredVolumeView.hpp"
#include "StructuredVolume_impl.hpp"
#include "TransfuncEditor.hpp"

using namespace visionaray;

#if VSNRAY_COMMON_HAVE_SDL2
using ViewerBase = viewer_sdl2;
#else
using ViewerBase = viewer_glut;
#endif

//-------------------------------------------------------------------------------------------------
// I/O utility for camera lookat only - not fit for the general case!
//

inline mat3 matrixFromAxisAngle(vec3 a1, float angle)
{

    double c = cos(angle);
    double s = sin(angle);
    double t = 1.0 - c;
    //  if axis is not already normalised then uncomment this
    // double magnitude = Math.sqrt(a1.x*a1.x + a1.y*a1.y + a1.z*a1.z);
    // if (magnitude==0) throw error;
    // a1.x /= magnitude;
    // a1.y /= magnitude;
    // a1.z /= magnitude;

    float m00 = c + a1.x * a1.x * t;
    float m11 = c + a1.y * a1.y * t;
    float m22 = c + a1.z * a1.z * t;

    double tmp1 = a1.x * a1.y * t;
    double tmp2 = a1.z * s;
    float m10 = tmp1 + tmp2;
    float m01 = tmp1 - tmp2;
    tmp1 = a1.x * a1.z * t;
    tmp2 = a1.y * s;
    float m20 = tmp1 - tmp2;
    float m02 = tmp1 + tmp2;
    tmp1 = a1.y * a1.z * t;
    tmp2 = a1.x * s;
    float m21 = tmp1 + tmp2;
    float m12 = tmp1 - tmp2;
    mat3 res = mat3(
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22);
    return res;
}

inline std::istream &operator>>(std::istream &in, thin_lens_camera &cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

inline std::ostream &operator<<(std::ostream &out, thin_lens_camera const &cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}

//-------------------------------------------------------------------------------------------------
// Visionaray viewer
//
bool captured = false;
std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> elapsed_time;
const int MAX_SCREENSHOTS = 400;
const int TIME_BETWEEN_SCREENSHOTS = 5;
const bool prepareNoisyData = false;
struct Viewer : ViewerBase
{
    // using RayType = basic_ray<simd::float4>;
    using RayType = basic_ray<float>;

    vkt::StructuredVolume *structuredVolumes;
    vkt::HierarchicalVolume *hierarchicalVolumes;
    std::size_t numAnimationFrames;
    vkt::RenderState renderState;

    std::vector<vkt::StructuredVolumeView> structuredVolumeViews;
    std::vector<vkt::HierarchicalVolumeAccel> hierarchicalVolumeAccels;
    std::vector<vkt::HierarchicalVolumeView> hierarchicalVolumeViews;

    aabb bbox;
    matrix_camera cam;
    mat4 view;
    mat4 proj;
    unsigned frame_num;

    vkt::TransfuncEditor transfuncEditor;

    std::future<void> renderFuture;
    std::mutex displayMutex;

    int frontBufferIndex;

    bool useCuda;
    float rotation_factor = 0.5f;
    int num_screenshots = 0;
    // Two render targets for double buffering
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> host_rt[2];
    tiled_sched<RayType> host_sched;
    std::vector<vec4> host_accumBuffer;
    std::vector<vec4> host_accumAlbedoBuffer;
    std::vector<vec4> host_accumPositionBuffer;
    std::vector<vec4> host_accumGradientBuffer;
    std::vector<vec4> host_accumSecondAlbedoBuffer;
    std::vector<vec4> host_accumSecondCharacteristicsBuffer;
    std::vector<vec4> host_accumSecondGradientBuffer;
    std::vector<vec4> host_accumCharacteristicsBuffer;
#if VKT_HAVE_CUDA
    // Two render targets for double buffering
    gpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> device_rt[2];
    cuda_sched<RayType> device_sched;
    thrust::device_vector<vec4> device_accumBuffer;
    thrust::device_vector<vec4> device_accumAlbedoBuffer;
    thrust::device_vector<vec4> device_accumPositionBuffer;
    thrust::device_vector<vec4> device_accumGradientBuffer;
    thrust::device_vector<vec4> device_accumSecondAlbedoBuffer;
    thrust::device_vector<vec4> device_accumSecondCharacteristicsBuffer;
    thrust::device_vector<vec4> device_accumSecondGradientBuffer;
    thrust::device_vector<vec4> device_accumCharacteristicsBuffer;
    cuda_texture<int16_t, 3> device_volumeInt16;
    cuda_texture<uint8_t, 3> device_volumeUint8;
    cuda_texture<uint16_t, 3> device_volumeUint16;
    cuda_texture<uint32_t, 3> device_volumeUint32;
    cuda_texture<float, 3> device_volumeFloat32;
    cuda_texture<vec4, 1> device_transfunc;

    inline cuda_texture_ref<int16_t, 3> prepareDeviceVolume(int16_t /* */)
    {
        return cuda_texture_ref<int16_t, 3>(device_volumeInt16);
    }

    inline cuda_texture_ref<uint8_t, 3> prepareDeviceVolume(uint8_t /* */)
    {
        return cuda_texture_ref<uint8_t, 3>(device_volumeUint8);
    }

    inline cuda_texture_ref<uint16_t, 3> prepareDeviceVolume(uint16_t /* */)
    {
        return cuda_texture_ref<uint16_t, 3>(device_volumeUint16);
    }

    inline cuda_texture_ref<uint32_t, 3> prepareDeviceVolume(uint32_t /* */)
    {
        return cuda_texture_ref<uint32_t, 3>(device_volumeUint32);
    }

    inline cuda_texture_ref<float, 3> prepareDeviceVolume(float /* */)
    {
        return cuda_texture_ref<float, 3>(device_volumeFloat32);
    }

    inline cuda_texture_ref<vec4, 1> prepareDeviceTransfunc()
    {
        using namespace vkt;

        if (renderState.rgbaLookupTable != ResourceHandle(-1))
        {
            LookupTable *lut = transfuncEditor.getUpdatedLookupTable();
            if (lut == nullptr)
                lut = (LookupTable *)GetManagedResource(renderState.rgbaLookupTable);

            if (transfuncEditor.updated() || !device_transfunc)
            {
                ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();
                ExecutionPolicy prev = ep;
                ep.device = vkt::ExecutionPolicy::Device::CPU;
                SetThreadExecutionPolicy(ep);

                device_transfunc = cuda_texture<vec4, 1>(
                    (vec4 *)lut->getData(),
                    lut->getDims().x,
                    Clamp,
                    Nearest);

                SetThreadExecutionPolicy(prev);
            }

            return cuda_texture_ref<vec4, 1>(device_transfunc);
        }
        else
            return cuda_texture_ref<vec4, 1>();
    }
#endif

    Viewer(
        vkt::StructuredVolume *structuredVolumes,
        vkt::HierarchicalVolume *hierarchicalVolumes,
        std::size_t numAnimationFrames,
        vkt::RenderState renderState,
        char const *windowTitle = "",
        unsigned numThreads = std::thread::hardware_concurrency());

    void createVolumeViews();

    void updateVolumeTexture();

    void clearFrame();

    void screenShot();
    void captureRGB();
    void captureAlbedo();
    void capturePosition();
    void captureGradient();
    void captureSecondAlbedo();
    void captureSecondCharacteristics();
    void captureSecondGradient();
    void captureCharacteristics();
    void switchView();
    void on_display();
    void resetVectors();
    void on_key_press(visionaray::key_event const &event);
    void on_mouse_move(visionaray::mouse_event const &event);
    void on_space_mouse_move(visionaray::space_mouse_event const &event);
    void on_resize(int w, int h);

};

Viewer::Viewer(
    vkt::StructuredVolume *structuredVolumes,
    vkt::HierarchicalVolume *hierarchicalVolumes,
    std::size_t numAnimationFrames,
    vkt::RenderState renderState,
    char const *windowTitle,
    unsigned numThreads)
    : ViewerBase(renderState.viewportWidth, renderState.viewportHeight, windowTitle), structuredVolumes(structuredVolumes), hierarchicalVolumes(hierarchicalVolumes), numAnimationFrames(numAnimationFrames), renderState(renderState), host_sched(numThreads), frontBufferIndex(0)
{
    vkt::ExecutionPolicy ep = vkt::GetThreadExecutionPolicy();

    useCuda = ep.device == vkt::ExecutionPolicy::Device::GPU && ep.deviceApi == vkt::ExecutionPolicy::DeviceAPI::CUDA;

    if (renderState.rgbaLookupTable != vkt::ResourceHandle(-1))
        transfuncEditor.setLookupTableResource(renderState.rgbaLookupTable);

    if (renderState.histogram != vkt::ResourceHandle(-1))
        transfuncEditor.setHistogramResource(renderState.histogram);

    createVolumeViews();

    updateVolumeTexture();
    elapsed_time = std::chrono::system_clock::now();
}

void Viewer::createVolumeViews()
{
    if (structuredVolumes != nullptr)
    {
        structuredVolumeViews.resize(numAnimationFrames);
        for (std::size_t i = 0; i < numAnimationFrames; ++i)
        {
            structuredVolumeViews[i] = vkt::StructuredVolumeView(structuredVolumes[i]);
        }
    }
    else if (hierarchicalVolumes != nullptr)
    {
        hierarchicalVolumeAccels.resize(numAnimationFrames);
        hierarchicalVolumeViews.resize(numAnimationFrames);
        for (std::size_t i = 0; i < numAnimationFrames; ++i)
        {
            hierarchicalVolumeAccels[i] = vkt::HierarchicalVolumeAccel(hierarchicalVolumes[i]);
            hierarchicalVolumeViews[i] = vkt::HierarchicalVolumeView(
                hierarchicalVolumes[i],
                hierarchicalVolumeAccels[i]);
        }
    }
}

void Viewer::updateVolumeTexture()
{
    if (structuredVolumes == nullptr)
        return;

    vkt::StructuredVolumeView volume = structuredVolumeViews[renderState.animationFrame];

    // Initialize device textures
    if (useCuda)
    {
#if VKT_HAVE_CUDA
        switch (volume.getDataFormat())
        {
        case vkt::DataFormat::Int16:
            device_volumeInt16 = cuda_texture<int16_t, 3>(
                (int16_t *)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest);
            break;
        case vkt::DataFormat::UInt8:
            device_volumeUint8 = cuda_texture<uint8_t, 3>(
                (uint8_t *)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest);
            break;
        case vkt::DataFormat::UInt16:
            device_volumeUint16 = cuda_texture<uint16_t, 3>(
                (uint16_t *)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest);
            break;
        case vkt::DataFormat::UInt32:
            device_volumeUint32 = cuda_texture<uint32_t, 3>(
                (uint32_t *)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest);
            break;
        case vkt::DataFormat::Float32:
            device_volumeFloat32 = cuda_texture<float, 3>(
                (float *)volume.getData(),
                volume.getDims().x,
                volume.getDims().y,
                volume.getDims().z,
                Clamp,
                Nearest);
            break;
        }
#else
        VKT_LOG(vkt::logging::Level::Error) << " GPU backend not available";
#endif
    }
}

void Viewer::clearFrame()
{
    std::unique_lock<std::mutex> l(displayMutex);
    frame_num = 0;
}

void Viewer::screenShot()
{
    auto const &rt = host_rt[frontBufferIndex];

    // Swizzle to RGB8 for compatibility with pnm image
    std::vector<vector<3, unorm<8>>> rgb(rt.width() * rt.height());

    glReadPixels(0, 0, rt.width(), rt.height(), GL_RGB, GL_UNSIGNED_BYTE, rgb.data());

    // Flip so that origin is (top|left)
    std::vector<vector<3, unorm<8>>> flipped(rgb.size());

    for (int y = 0; y < rt.height(); ++y)
    {
        for (int x = 0; x < rt.width(); ++x)
        {
            int yy = rt.height() - y - 1;
            flipped[yy * rt.width() + x] = rgb[y * rt.width() + x];
        }
    }

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(flipped.data()));

    image::save_option opt1;
    if (img.save(renderState.snapshotTool.fileName, {opt1}))
    {
        std::string message(renderState.snapshotTool.message);
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureRGB()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());
    std::string screenshotName = "";
    if (prepareNoisyData)
    {
        screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/rgb/";
    }
    else
    {
        screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/groundTruth/";
    }

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append(".hdr");

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}

void Viewer::capturePosition()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumPositionBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/position/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_pos.hdr");

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureCharacteristics()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumCharacteristicsBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/characteristics/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_vol1.hdr");
    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureSecondCharacteristics()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumSecondCharacteristicsBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/secondCharacteristics/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_vol2.hdr");

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureAlbedo()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumAlbedoBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/albedo/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_alb1.hdr");

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureSecondAlbedo()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumSecondAlbedoBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/secondAlbedo/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_alb2.hdr");

    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}

void Viewer::captureGradient()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumGradientBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/gradient/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_norm1.hdr");
    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}
void Viewer::captureSecondGradient()
{
    auto const &rt = host_rt[frontBufferIndex];
    // albedobuffer is of type thrust::device_vector<vec3>
    thrust::host_vector<vec4> h_v(device_accumSecondGradientBuffer);
    std::vector<vector<3, unorm<8>>> output(h_v.begin(), h_v.end());

    std::string screenshotName = "/home/niklas/Dokumente/discovering-the-impact-of-volume-path-tracing-denoisers-on-features-in-medical-data/dataset/1spp/secondGradient/";

    screenshotName.append(std::to_string(num_screenshots));
    screenshotName.append("_norm2.hdr");
    image img(
        rt.width(),
        rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const *>(output.data()));

    image::save_option opt1;

    if (img.save(screenshotName.c_str(), {opt1}))
    {
        std::string message(screenshotName);
        message.append(" saved");
        if (!message.empty())
            std::cout << message << '\n';
    }
    else
    {
        VKT_LOG(vkt::logging::Level::Error) << " Error taking screen shot";
    }
}

void Viewer::resetVectors()
{
#if VKT_HAVE_CUDA
    thrust::fill(device_accumAlbedoBuffer.begin(), device_accumAlbedoBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumPositionBuffer.begin(), device_accumPositionBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumGradientBuffer.begin(), device_accumGradientBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumCharacteristicsBuffer.begin(), device_accumCharacteristicsBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumSecondAlbedoBuffer.begin(), device_accumSecondAlbedoBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumSecondGradientBuffer.begin(), device_accumSecondGradientBuffer.end(), vec4(0, 0, 0, 0));
    thrust::fill(device_accumSecondCharacteristicsBuffer.begin(), device_accumSecondCharacteristicsBuffer.end(), vec4(0, 0, 0, 0));
#endif
}
void Viewer::on_display()
{
    if (transfuncEditor.updated())
        clearFrame();

    vkt::StructuredVolumeView structuredVolume;
    vkt::HierarchicalVolumeView hierarchicalVolume;

    bool structured = structuredVolumes != nullptr;

    if (structured)
        structuredVolume = structuredVolumeViews[renderState.animationFrame];
    else
        hierarchicalVolume = hierarchicalVolumeViews[renderState.animationFrame];

    // Prepare a kernel with the volume set up appropriately
    // according to the provided texture and texel type
    auto prepareStructuredVolume = [&](auto texel)
    {
        using TexelType = decltype(texel);
        using Texture = texture_ref<TexelType, 3>;

        Texture volume_tex(
            structuredVolume.getDims().x,
            structuredVolume.getDims().y,
            structuredVolume.getDims().z);
        volume_tex.reset((TexelType *)structuredVolume.getData());
        volume_tex.set_filter_mode(Nearest);
        volume_tex.set_address_mode(Clamp);
        return volume_tex;
    };

    auto prepareTransfunc = [&]()
    {
        using namespace vkt;

        texture_ref<vec4, 1> transfunc_tex(0U);

        if (renderState.rgbaLookupTable != ResourceHandle(-1))
        {
            LookupTable *lut = transfuncEditor.getUpdatedLookupTable();
            if (lut == nullptr)
                lut = (LookupTable *)GetManagedResource(renderState.rgbaLookupTable);

            transfunc_tex = texture_ref<vec4, 1>(lut->getDims().x);
            transfunc_tex.set_filter_mode(Nearest);
            transfunc_tex.set_address_mode(Clamp);
            transfunc_tex.reset((vec4 *)lut->getData());
        }

        return transfunc_tex;
    };

    auto prepareRayMarchingKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        RayMarchingKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.dt = renderState.dtRayMarching;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto prepareImplicitIsoKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        ImplicitIsoKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.numIsoSurfaces = renderState.numIsoSurfaces;
        std::memcpy(
            &kernel.isoSurfaces,
            &renderState.isoSurfaces,
            sizeof(renderState.isoSurfaces));
        kernel.dt = renderState.dtImplicitIso;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto prepareMultiScatteringKernel = [&](auto volume_tex, auto transfunc_tex, auto accumBuffer, auto accumAlbedoBuffer, auto accumPositionBuffer, auto accumGradientBuffer,
                                            auto accumCharacteristicsBuffer, auto accumSecondAlbedoBuffer, auto accumSecondGradientBuffer, auto accumSecondCharacteristicsBuffer)
    {
        using VolumeTex = decltype(volume_tex);
        using TransfuncTex = decltype(transfunc_tex);

        float heightf(this->height());
        MultiScatteringKernel<VolumeTex, TransfuncTex> kernel;
        kernel.bbox = bbox;
        kernel.volume = volume_tex;
        kernel.transfunc = transfunc_tex;
        kernel.mu_ = renderState.majorant;
        kernel.heightf_ = heightf;
        kernel.width = width();
        kernel.height = height();
        kernel.frameNum = frame_num;
        kernel.accumBuffer = accumBuffer;
        kernel.accumAlbedoBuffer = accumAlbedoBuffer;
        kernel.accumPositionBuffer = accumPositionBuffer;
        kernel.accumGradientBuffer = accumGradientBuffer;
        kernel.accumCharacteristicsBuffer = accumCharacteristicsBuffer;
        kernel.accumSecondAlbedoBuffer = accumSecondAlbedoBuffer;
        kernel.accumSecondGradientBuffer = accumSecondGradientBuffer;
        kernel.accumSecondCharacteristicsBuffer = accumSecondCharacteristicsBuffer;
        kernel.sRGB = (bool)renderState.sRGB;

        return kernel;
    };

    auto callKernel = [&](auto texel)
    {
        using TexelType = decltype(texel);

        if (!renderFuture.valid() || renderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            {
                std::unique_lock<std::mutex> l(displayMutex);
                // swap render targets
                frontBufferIndex = !frontBufferIndex;

                ++frame_num;
            }

            vkt::ExecutionPolicy mainThreadEP = vkt::GetThreadExecutionPolicy();

            renderFuture = std::async(
                [&, mainThreadEP, this]()
                {
                    vkt::SetThreadExecutionPolicy(mainThreadEP);

                    if (useCuda)
                    {
#if VKT_HAVE_CUDA
                        pixel_sampler::jittered_type blend_params;
                        auto sparams = make_sched_params(
                            blend_params,
                            view,
                            proj,
                            device_rt[!frontBufferIndex]);

                        if (renderState.renderAlgo == vkt::RenderAlgo::RayMarching)
                        {
                            auto kernel = prepareRayMarchingKernel(
                                prepareDeviceVolume(TexelType{}),
                                prepareDeviceTransfunc(),
                                thrust::raw_pointer_cast(device_accumBuffer.data()));
                            device_sched.frame(kernel, sparams);
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::ImplicitIso)
                        {
                            auto kernel = prepareImplicitIsoKernel(
                                prepareDeviceVolume(TexelType{}),
                                prepareDeviceTransfunc(),
                                thrust::raw_pointer_cast(device_accumBuffer.data()));
                            device_sched.frame(kernel, sparams);
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::MultiScattering)
                        {

                            if (structured)
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                    prepareDeviceVolume(TexelType{}),
                                    prepareDeviceTransfunc(),
                                    thrust::raw_pointer_cast(device_accumBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumAlbedoBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumPositionBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumGradientBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumCharacteristicsBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondAlbedoBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondGradientBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondCharacteristicsBuffer.data()));
                                device_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                    hierarchicalVolume,
                                    prepareDeviceTransfunc(),
                                    thrust::raw_pointer_cast(device_accumBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumAlbedoBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumPositionBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumGradientBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumCharacteristicsBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondAlbedoBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondGradientBuffer.data()),
                                    thrust::raw_pointer_cast(device_accumSecondCharacteristicsBuffer.data()));
                                device_sched.frame(kernel, sparams);
                            }
                        }
#else
                        VKT_LOG(vkt::logging::Level::Error)
                            << " GPU backend not available";
#endif
                    }
                    else
                    {
#ifndef __CUDA_ARCH__
                        pixel_sampler::jittered_type blend_params;
                        auto sparams = make_sched_params(
                            blend_params,
                            view,
                            proj,
                            host_rt[!frontBufferIndex]);

                        if (renderState.renderAlgo == vkt::RenderAlgo::RayMarching)
                        {
                            if (structured)
                            {
                                auto kernel = prepareRayMarchingKernel(
                                    prepareStructuredVolume(TexelType{}),
                                    prepareTransfunc(),
                                    host_accumBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareRayMarchingKernel(
                                    hierarchicalVolume,
                                    prepareTransfunc(),
                                    host_accumBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::ImplicitIso)
                        {
                            if (structured)
                            {
                                auto kernel = prepareImplicitIsoKernel(
                                    prepareStructuredVolume(TexelType{}),
                                    prepareTransfunc(),
                                    host_accumBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareImplicitIsoKernel(
                                    hierarchicalVolume,
                                    prepareTransfunc(),
                                    host_accumBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                        }
                        else if (renderState.renderAlgo == vkt::RenderAlgo::MultiScattering)
                        {
                            if (structured)
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                    prepareStructuredVolume(TexelType{}),
                                    prepareTransfunc(),
                                    host_accumBuffer.data(),
                                    host_accumAlbedoBuffer.data(),
                                    host_accumPositionBuffer.data(),
                                    host_accumGradientBuffer.data(),
                                    host_accumCharacteristicsBuffer.data(),
                                    host_accumSecondAlbedoBuffer.data(),
                                    host_accumSecondGradientBuffer.data(),
                                    host_accumSecondCharacteristicsBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                            else
                            {
                                auto kernel = prepareMultiScatteringKernel(
                                    hierarchicalVolume,
                                    prepareTransfunc(),
                                    host_accumBuffer.data(),
                                    host_accumAlbedoBuffer.data(),
                                    host_accumPositionBuffer.data(),
                                    host_accumGradientBuffer.data(),
                                    host_accumCharacteristicsBuffer.data(),
                                    host_accumSecondAlbedoBuffer.data(),
                                    host_accumSecondGradientBuffer.data(),
                                    host_accumSecondCharacteristicsBuffer.data());
                                host_sched.frame(kernel, sparams);
                            }
                        }
#endif
                    }
                });
        }
    };
   

            if (structured)
            {
                switch (structuredVolume.getDataFormat())
                {
                case vkt::DataFormat::Int16:
                    callKernel(int16_t{});
                    break;
                case vkt::DataFormat::UInt8:
                    callKernel(uint8_t{});
                    break;
                case vkt::DataFormat::UInt16:
                    callKernel(uint16_t{});
                    break;
                case vkt::DataFormat::UInt32:
                    callKernel(uint32_t{});
                    break;
                case vkt::DataFormat::Float32:
                    callKernel(float{});
                    break;
                }
            }
            else // hierarchical
            {
                callKernel(uint8_t{});
            }
        

    // display the rendered image

    auto bgcolor = background_color();

    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
        std::unique_lock<std::mutex> l(displayMutex);
        if (useCuda)
        {
#if VKT_HAVE_CUDA
            device_rt[frontBufferIndex].display_color_buffer();
#else
            VKT_LOG(vkt::logging::Level::Error) << " GPU backend not available";
#endif
        }
        else
            host_rt[frontBufferIndex].display_color_buffer();
    }

    if (have_imgui_support() && renderState.rgbaLookupTable != vkt::ResourceHandle(-1))
        transfuncEditor.show();
    // just a dirty hack to delay the time to make screenshots
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - elapsed_time;
    if (elapsed_seconds.count() > TIME_BETWEEN_SCREENSHOTS)
    {
        if (num_screenshots == MAX_SCREENSHOTS)
        {
            viewer_glut::quit();
        }
        switchView();
    }
}
void Viewer::switchView()
{
    captureRGB();
    if (prepareNoisyData)
    {
        captureAlbedo();
        capturePosition();
        captureGradient();
        captureCharacteristics();
        captureSecondAlbedo();
        captureSecondCharacteristics();
        captureSecondGradient();
    }

    num_screenshots++;
    // vec3 up(0, 1, 0);
    // // float diagonal = length(bbox.size());
    // // float r = diagonal * 0.5f;

    // // float dotsurfeye = dot(vec3(0,0,1), cam.eye());
    // vec3 eye(0, 0, 0);
    // vec3 right = normalize(cross(cam.eye(), up));
    // vec3 camFocusVector(cam.eye() - bbox.center());
    // mat3 rotMat1 = matrixFromAxisAngle(up, rotation_factor);
    // mat3 rotMat2 = matrixFromAxisAngle(right, rotation_factor);
    // camFocusVector = rotMat1 * camFocusVector;
    // camFocusVector = rotMat2 * camFocusVector;
    // eye = camFocusVector + bbox.center();
    // cam.look_at(eye, bbox.center(), up);

    // renderState.initialCamera.eye = {eye.x, eye.y, eye.z};
    // renderState.initialCamera.center = {bbox.center().x, bbox.center().y, bbox.center().z};
    // renderState.initialCamera.up = {up.x, up.y, up.z};
    // // updateVolumeTexture();
    
    clearFrame();

    elapsed_time = std::chrono::system_clock::now();
    if (num_screenshots % 20 == 0)
    {
        renderState.animationFrame++;

        renderState.animationFrame %= numAnimationFrames;
        updateVolumeTexture();
    }
    resetVectors();
}
void Viewer::on_key_press(visionaray::key_event const &event)
{
    if (event.key() == keyboard::Space)
    {
        renderState.animationFrame++;
        renderState.animationFrame %= numAnimationFrames;
        updateVolumeTexture();
        clearFrame();
    }

    if (renderState.snapshotTool.enabled && event.key() == renderState.snapshotTool.key)
    {
        screenShot();
    }

    ViewerBase::on_key_press(event);
}

void Viewer::on_mouse_move(visionaray::mouse_event const &event)
{
    if (event.buttons() != mouse::NoButton)
        clearFrame();

    ViewerBase::on_mouse_move(event);
}

void Viewer::on_space_mouse_move(visionaray::space_mouse_event const &event)
{
    clearFrame();

    ViewerBase::on_space_mouse_move(event);
}

void Viewer::on_resize(int w, int h)
{
    if (renderFuture.valid())
        renderFuture.wait();

    // cam.set_viewport(0, 0, w, h);
    // float aspect = w / static_cast<float>(h);
    // cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    {
        std::unique_lock<std::mutex> l(displayMutex);

        host_accumBuffer.resize(w * h);
        host_accumAlbedoBuffer.resize(w * h);
        host_accumPositionBuffer.resize(w * h);
        host_accumGradientBuffer.resize(w * h);
        host_accumSecondAlbedoBuffer.resize(w * h);
        host_accumSecondCharacteristicsBuffer.resize(w * h);
        host_accumSecondGradientBuffer.resize(w * h);
        host_accumCharacteristicsBuffer.resize(w * h);
        host_rt[0].resize(w, h);
        host_rt[1].resize(w, h);

#if VKT_HAVE_CUDA
        device_accumAlbedoBuffer.resize(w * h);
        device_accumPositionBuffer.resize(w * h);
        device_accumBuffer.resize(w * h);
        device_accumGradientBuffer.resize(w * h);
        device_accumSecondAlbedoBuffer.resize(w * h);
        device_accumSecondCharacteristicsBuffer.resize(w * h);
        device_accumSecondGradientBuffer.resize(w * h);
        device_accumCharacteristicsBuffer.resize(w * h);
        device_rt[0].resize(w, h);
        device_rt[1].resize(w, h);
#endif
    }

    clearFrame();

    ViewerBase::on_resize(w, h);
}

//-------------------------------------------------------------------------------------------------
// Render common impl for both APIs
//

static void Render_impl(
    vkt::StructuredVolume *structuredVolumes,
    vkt::HierarchicalVolume *hierarchicalVolumes,
    std::size_t numAnimationFrames,
    vkt::RenderState const &renderState,
    vkt::RenderState *newRenderState)
{
    Viewer viewer(structuredVolumes, hierarchicalVolumes, numAnimationFrames, renderState);

    int argc = 1;
    char const *argv = "vktRender";
    viewer.init(argc, (char **)&argv);

    vkt::Vec3i dims;
    vkt::Vec3f dist{1.f, 1.f, 1.f};

    if (structuredVolumes != nullptr)
    {
        dims = structuredVolumes[0].getDims();
        dist = structuredVolumes[0].getDist();
    }
    else if (hierarchicalVolumes != nullptr)
        dims = hierarchicalVolumes[0].getDims();

    viewer.bbox = aabb(
        {0.f, 0.f, 0.f},
        {dims.x * dist.x, dims.y * dist.y, dims.z * dist.z});

    float aspect = viewer.width() / static_cast<float>(viewer.height());

    float fovy = 45.f * constants::degrees_to_radians<float>();

    float diagonal = length(viewer.bbox.size());
    float r = diagonal *0.5f;
    vec3 eye = viewer.bbox.center() + vec3(0, 0, r +r /std::atan(fovy));
    
    
    
    
    vec3 up = vec3(0.0f,1.0f,0.0f);
    vec3 f = normalize(eye - viewer.bbox.center());
    vec3 s = normalize(cross(up, f));
    vec3 u = cross(f, s);


    float z_near = 0.001f;
    float z_far = 1000.0f;
    viewer.view = mat4(
        s.x, u.x, f.x, 0.0f,
        s.y, u.y, f.y, 0.0f,
        s.z, u.z, f.z, 0.0f,
        -dot(eye, s), -dot(eye, u), -dot(eye, f), 1.0f
        );


 
   

    

    viewer.proj(0, 0) = 1.0f / (viewer.width()/2.0f);
    viewer.proj(0, 1) = 0.0f;
    viewer.proj(0, 2) = 0.0f;
    viewer.proj(0, 3) = 0.0f;
    
    viewer.proj(1, 0) = 0.0f;
    viewer.proj(1, 1) = 1.0f / (viewer.height()/2.0f);
    viewer.proj(1, 2) = 0.0f;
    viewer.proj(1, 3) = 0.0f;

    viewer.proj(2, 0) = 0.0f;
    viewer.proj(2, 1) = 0.0f;
    viewer.proj(2, 2) = -2.0f / (z_far - z_near);
    viewer.proj(2, 3) = -((z_far + z_near) / (z_far - z_near));

    viewer.proj(3, 0) = 0.0f;
    viewer.proj(3, 1) = 0.0f;
    viewer.proj(3, 2) = 0.0f;
    viewer.proj(3, 3) = 1.0f;
    
    viewer.event_loop();

   
}

//-------------------------------------------------------------------------------------------------
// Overloads for single volume
//

static void Render_impl(
    vkt::StructuredVolume &volume,
    vkt::RenderState const &renderState,
    vkt::RenderState *newRenderState)
{
    Render_impl(&volume, nullptr, 1, renderState, newRenderState);
}

static void Render_impl(
    vkt::HierarchicalVolume &volume,
    vkt::RenderState const &renderState,
    vkt::RenderState *newRenderState)
{
    Render_impl(nullptr, &volume, 1, renderState, newRenderState);
}

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace vkt
{

    Error Render(StructuredVolume &volume, RenderState const &renderState, RenderState *newRenderState)
    {
        Render_impl(volume, renderState, newRenderState);

        return NoError;
    }

    Error RenderFrames(
        StructuredVolume *frames,
        std::size_t numAnimationFrames,
        RenderState const &renderState,
        RenderState *newRenderState)
    {
        Render_impl(frames, nullptr, numAnimationFrames, renderState, newRenderState);

        return NoError;
    }

    Error Render(HierarchicalVolume &volume, RenderState const &renderState, RenderState *newRenderState)
    {
        Render_impl(volume, renderState, newRenderState);

        return NoError;
    }

    Error RenderFrames(
        HierarchicalVolume *frames,
        std::size_t numAnimationFrames,
        RenderState const &renderState,
        RenderState *newRenderState)
    {
        Render_impl(nullptr, frames, numAnimationFrames, renderState, newRenderState);

        return NoError;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
//

vktError vktRenderSV(
    vktStructuredVolume volume,
    vktRenderState_t renderState,
    vktRenderState_t *newRenderState)
{
    static_assert(sizeof(vktRenderState_t) == sizeof(vkt::RenderState), "Type mismatch");

    vkt::RenderState renderStateCPP;

    std::memcpy(&renderStateCPP, &renderState, sizeof(renderState));

    vkt::RenderState newRenderStateCPP;
    Render_impl(volume->volume, renderStateCPP, &newRenderStateCPP);

    if (newRenderState != nullptr)
        std::memcpy(newRenderState, &newRenderStateCPP, sizeof(newRenderStateCPP));

    return vktNoError;
}