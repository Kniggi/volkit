// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cassert>

#include <visionaray/math/aabb.h>
#include <visionaray/math/intersect.h>
#include <visionaray/math/limits.h>
#include <visionaray/math/vector.h>
#include <visionaray/phase_function.h>
#include <visionaray/result_record.h>
#include <common/image.h>
#include <vkt/Render.hpp>

#include "HierarchicalVolumeView.hpp"


VKT_FUNC inline float tex3D(vkt::HierarchicalVolumeView const& vol, visionaray::vec3 coord)
{
    assert(vol.getDataFormat() == vkt::DataFormat::Float32);

    visionaray::vec3 logicalGridDims((float)vol.getDims().x, (float)vol.getDims().y, (float)vol.getDims().z);

    coord *= logicalGridDims-visionaray::vec3(.5f);

    return vol.sampleLinear(coord.x, coord.y, coord.z);
}

template <typename Volume>
VKT_FUNC float normalize(Volume /* */, float voxel)
{
    typename Volume::value_type minval = visionaray::numeric_limits<typename Volume::value_type>::min();
    typename Volume::value_type maxval = visionaray::numeric_limits<typename Volume::value_type>::max();
    voxel -= (float)minval;
    voxel /= (float)(maxval - minval);
    return voxel;
}

VKT_FUNC inline float normalize(vkt::HierarchicalVolumeView const& volume, float voxel)
{
    assert(volume.getDataFormat() == vkt::DataFormat::Float32);

    voxel -= volume.getVoxelMapping().x;
    voxel /= volume.getVoxelMapping().y - volume.getVoxelMapping().x;
    return voxel;
}


struct AccumulationKernel
{
    int width;
    int height;
    unsigned frameNum;
    bool sRGB;
    visionaray::vec4f* accumBuffer = nullptr;
    visionaray::vec3f* albedoBuffer = nullptr;

    VSNRAY_FUNC
    visionaray::vec4f accum(visionaray::vec4f src, int x, int y)
    {
        using namespace visionaray;
        //accumBuffer is where all the buffers come in
        float alpha = 1.f / frameNum;
        
        //albedoBuffer[y * width + x] = (1.f - alpha)* albedoBuffer[y * width + x] + alpha * albedo(src);
        accumBuffer[y * width + x] = (1.f - alpha) * accumBuffer[y * width + x] + alpha * src;
        vec4f result = accumBuffer[y * width + x];
    //    image img(
   //     width,
   //     height,
   //     PF_RGB8,
   //     reinterpret_cast<uint8_t const*>(accumBuffer)
   //     );
      //  image::save_option opt1;
       // img.save("test1.png",{opt1});
        if (sRGB)
            result.xyz() = linear_to_srgb(result.xyz());
        return result;
    }
};


//-------------------------------------------------------------------------------------------------
// Ray marching with absorption plus emission model
//

template <typename Volume, typename Transfunc>
struct RayMarchingKernel : AccumulationKernel
{
    template <typename Ray>
    VSNRAY_FUNC
    auto operator()(Ray ray, visionaray::random_generator<float>& gen, int x, int y)
    {
        using namespace visionaray;

        using S = typename Ray::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        vector<3, S> boxSize(bbox.size());
        vector<3, S> pos = ray.ori + ray.dir * t;
        vector<3, S> tex_coord = pos / boxSize;

        vector<3, S> inc = ray.dir * S(dt) / boxSize;

        C dst(0.f);

        while (visionaray::any(t < hit_rec.tfar))
        {
            // sample volume
            S voxel = (float)tex3D(volume, tex_coord);

            // normalize to [0..1]
            voxel = normalize(volume, voxel);

            // classification
            C color;
            if (transfunc)
                color = tex1D(transfunc, voxel);
            else
                color = C(voxel);

            // opacity correction
            color.w = S(1.f) - pow(S(1.f) - color.w, S(dt));

            // premultiplied alpha
            color.xyz() *= color.w;

            // front-to-back alpha compositing
            dst += select(
                    t < hit_rec.tfar,
                    color * (1.0f - dst.w),
                    C(0.0)
                    );

            // early-ray termination - don't traverse w/o a contribution
            if (visionaray::all(result.color.w >= 0.999))
            {
                break;
            }

            // step on
            tex_coord += inc;
            t += dt;
        }

        result.color = accum(dst, x, y);
        result.hit = hit_rec.hit;
        return result;
    }

    visionaray::aabb bbox;
    Volume volume;
    Transfunc transfunc;
    float dt;
};


//-------------------------------------------------------------------------------------------------
// Implicit iso-surface rendering
//

template <typename Volume, typename Transfunc>
struct ImplicitIsoKernel : AccumulationKernel
{
    template <typename T>
    VSNRAY_FUNC
    inline visionaray::vector<3, T> gradient(visionaray::vector<3, T> tex_coord)
    {
        using namespace visionaray;

        vector<3, T> s1;
        vector<3, T> s2;

        float DELTA = 0.01f;

        s1.x = tex3D(volume, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f));
        s2.x = tex3D(volume, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f));
        s1.y = tex3D(volume, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f));
        s2.y = tex3D(volume, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f));
        s1.z = tex3D(volume, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA));
        s2.z = tex3D(volume, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA));

        return s2 - s1;
    }

    template <typename Ray>
    VSNRAY_FUNC
    auto operator()(Ray ray, visionaray::random_generator<float>& gen, int x, int y)
    {
        using namespace visionaray;

        using S = typename Ray::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        vector<3, S> boxSize(bbox.size());
        vector<3, S> pos = ray.ori + ray.dir * t;
        vector<3, S> tex_coord = pos / boxSize;

        vector<3, S> inc = ray.dir * S(dt) / boxSize;

        S last(-1e20f);

        S isoT(-1e20f);

        C dst(0.f);

        while (visionaray::any(t < hit_rec.tfar))
        {
            // sample volume
            S voxel = (float)tex3D(volume, tex_coord);

            // normalize to [0..1]
            voxel = normalize(volume, voxel);

            if (visionaray::any(last >= S(-1e10f)))
            {
                for (uint16_t i = 0; i < numIsoSurfaces; ++i)
                {
                    if ((last <= isoSurfaces[i] && voxel >= isoSurfaces[i])
                     || (last >= isoSurfaces[i] && voxel <= isoSurfaces[i]))
                    {
                        C color;
                        if (transfunc)
                            color = tex1D(transfunc, voxel);
                        else
                            color = C(voxel);
                        vector<3, S> albedo = color.xyz();

                        isoT = t;
                        vector<3, S> N = normalize(gradient(tex_coord));
                        vector<3, S> ka(S(.2f));
                        vector<3, S> kd(max(0.f, dot(N, -ray.dir)) * voxel);
                        dst = C(ka + albedo * kd, S(1.f));
                    }
                }
            }

            if (visionaray::all(isoT >= S(-1e10f)))
                break;

            // step on
            tex_coord += inc;
            t += dt;
            last = voxel;
        }

        result.color = accum(dst, x, y);
        result.hit = isoT >= S(-1e10f);
        return result;
    }

    visionaray::aabb bbox;
    Volume volume;
    Transfunc transfunc;
    uint16_t numIsoSurfaces;
    float isoSurfaces[vkt::RenderState::MaxIsoSurfaces];
    float dt;
};


//-------------------------------------------------------------------------------------------------
// Simple multi-scattering
// Loosely based on M. Raab: Ray Tracing Inhomogeneous Volumes, RTGems I (2019)
//

template <typename Volume, typename Transfunc>
struct MultiScatteringKernel : AccumulationKernel
{
    VSNRAY_FUNC
    visionaray::vec3 albedo(visionaray::vec3 const& pos)
    {
        using namespace visionaray;

        float voxel = (float)tex3D(volume, pos / bbox.size());

        // normalize to [0..1]
        voxel = normalize(volume, voxel);

        if (transfunc)
        {
            vec4f rgba = tex1D(transfunc, voxel);
            return rgba.xyz();
        }
        else
            return vec3(voxel);
    }

    VSNRAY_FUNC
    float mu(visionaray::vec3 const& pos)
    {
        using namespace visionaray;

        float voxel = (float)tex3D(volume, pos / bbox.size());

        // normalize to [0..1]
        voxel = normalize(volume, voxel);

        if (transfunc)
        {
            vec4f rgba = tex1D(transfunc, voxel);
            return rgba.w;
        }
        else
            return voxel;
    }

    template <typename Ray>
    VSNRAY_FUNC
    bool sample_interaction(Ray& r, float d, visionaray::random_generator<float>& gen)
    {
        using namespace visionaray;

        float t = 0.0f;
        vec3 pos;

        do
        {
            //woodcock tracking
            t -= log(1.0f - gen.next()) / mu_;

            pos = r.ori + r.dir * t;
            if (t >= d)
            {
                return false;
            }
        }
        while (mu(pos) < gen.next() * mu_);

        r.ori = pos;
        return true;
    }

    template <typename Ray>
    VSNRAY_FUNC
    auto operator()(Ray r, visionaray::random_generator<float>& gen, int x, int y)
    {
        using namespace visionaray;

        using S = typename Ray::scalar_type;
        using C = vector<4, S>;

        henyey_greenstein<float> f;
        f.g = 0.f; // isotropic

        result_record<S> result;

        vec3 throughput(1.f);

        auto hit_rec = intersect(r, bbox);

        if (visionaray::any(hit_rec.hit))
        {
            r.ori += r.dir * hit_rec.tnear;
            hit_rec.tfar -= hit_rec.tnear;

            unsigned bounce = 0;

            while (sample_interaction(r, hit_rec.tfar, gen))
            {
                // Is the path length exceeded?
                if (bounce++ >= 1024)
                {
                    throughput = vec3(0.0f);
                    break;
                }

                throughput *= albedo(r.ori);

                // Russian roulette absorption
                float prob = max_element(throughput);
                if (prob < 0.2f)
                {
                    if (gen.next() > prob)
                    {
                        throughput = vec3(0.0f);
                        break;
                    }
                    throughput /= prob;
                }

                // Sample phase function
                vec3 scatter_dir;
                float pdf;
                f.sample(-r.dir, scatter_dir, pdf, gen);
                r.dir = scatter_dir;

                hit_rec = intersect(r, bbox);
            }
        }

        // Look up the environment
        float t = y / heightf_;
        vec3 Ld = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
        vec3 L = Ld * throughput;

        result.color = accum(vec4(L, 1.f), x, y);
        result.hit = hit_rec.hit;
        return result;
    }

    visionaray::aabb bbox;
    Volume volume;
    Transfunc transfunc;
    float mu_;
    float heightf_;
};
