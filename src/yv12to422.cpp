/*
  yv12to422.cpp

  This file is part of YV12To422

  Copyright (C) 2015 OKA Motofumi

  Author: OKA Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/

#ifdef DEBUG
    #include <iostream>
#endif

#include <cstdint>
#include <cmath>
#include <malloc.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>
#include "avisynth.h"

#include "proc_to422.h"
#include "simd.h"


#define YV12TO422_VERSION "0.3.0"


using planar_to_packed = void (__stdcall *)(
    const int width, const int height, const uint8_t* srcpy,
    const uint8_t* srcpu, const uint8_t* srcpv, uint8_t* dstp,
    const int dst_rowsize, const int src_pitch_y, const int src_pitch_uv,
    const int dst_pitch);


template <typename T>
static void __stdcall
planar_to_yuy2(const int widthuv, const int height, const uint8_t* srcpy,
               const uint8_t* srcpu, const uint8_t* srcpv, uint8_t* dstp,
               const int dst_rowsize, const int src_pitch_y,
               const int src_pitch_uv, const int dst_pitch)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < widthuv; x += sizeof(T)) {
            T y0, y1, y2, y3, u0, u1, v0, v1;
            y0 = load_reg((T*)(srcpy + 2 * x));
            cvtepu8_epi16x2(y0, y1);
            y2 = load_reg((T*)(srcpy + 2 * x + sizeof(T)));
            cvtepu8_epi16x2(y2, y3);

            u0 = load_reg((T*)(srcpu + x));
            cvtepu8_epi16x2r(u0, u1);
            v0 = load_reg((T*)(srcpv + x));
            cvtepu8_epi16x2r(v0, v1);

            y0 = or_reg(y0, unpacklo_epi16(u0, v0));
            y1 = or_reg(y1, unpackhi_epi16(u0, v0));
            y2 = or_reg(y2, unpacklo_epi16(u1, v1));
            y3 = or_reg(y3, unpackhi_epi16(u1, v1));

            int dpos = 4 * x;
            stream_reg((T*)(dstp + dpos + 0 * sizeof(T)), y0);
            stream_reg((T*)(dstp + dpos + 1 * sizeof(T)), y1);
            stream_reg((T*)(dstp + dpos + 2 * sizeof(T)), y2);
            stream_reg((T*)(dstp + dpos + 3 * sizeof(T)), y3);
        }
        srcpy += src_pitch_y;
        srcpu += src_pitch_uv;
        srcpv += src_pitch_uv;
        dstp += dst_pitch;
    }

}


class YV12To422 : public GenericVideoFilter
{
    VideoInfo vi_src;
    VideoInfo vi_yv16;
    VideoInfo vi_yuy2;
    bool yuy2out;
    bool lshift;
    int dvpal;
    int memalign;
    int num_threads;
    int16_t cubic_coefficients[8];

    proc_to422 proc_chroma;
    proc_horizontal proc_chroma_qpel_shift_h;
    planar_to_packed yv16toyuy2;


public:
    YV12To422(
        PClip child, int itype, bool interlaced, int cplace, double _b,
        double _c, bool yuy2, bool avx2, bool lshift, int threads,
        IScriptEnvironment* env);
    ~YV12To422() {};
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


extern void set_cubic_coefficients(double b, double c, int16_t* array, bool interlaced, int cplace);


YV12To422::
YV12To422(PClip _child, int itype, bool interlaced, int cplace, double b,
           double c, bool yuy2, bool avx2, bool _lshift, int threads,
           IScriptEnvironment* env)
  : GenericVideoFilter(_child),
    yuy2out(yuy2),
    lshift(_lshift),
    num_threads(threads)
{
    memalign = avx2 ? sizeof(__m256i) : sizeof(__m128i);
    proc_chroma = get_proc_chroma(itype, cplace, interlaced, avx2);
    proc_chroma_qpel_shift_h = get_proc_horizontal_shift(avx2);
    yv16toyuy2 = avx2 ? planar_to_yuy2<__m256i> : planar_to_yuy2<__m128i>;
    if (itype == 2) {
        set_cubic_coefficients(b, c, cubic_coefficients, interlaced, cplace);
    }

    dvpal = interlaced && cplace == 3 ? -1 : 1;

    memcpy(&vi_src, &vi, sizeof(VideoInfo));
    memcpy(&vi_yv16, &vi, sizeof(VideoInfo));
    vi_yv16.pixel_type = VideoInfo::CS_YV16;
    if (yuy2out) {
        vi.pixel_type = VideoInfo::CS_YUY2;
        memcpy(&vi_yuy2, &vi, sizeof(VideoInfo));
        vi_yuy2.width = aligned_size(vi.width, memalign * 2);
    } else {
        vi.pixel_type = VideoInfo::CS_YV16;
    }

#ifdef DEBUG
    std::cerr << "cplace:" << cplace << " itype:" << itype << " interlaced:"
        << interlaced << " yuy2:" << yuy2out << " avx2:" << avx2 <<
        " threads: " << threads << "\n";
#endif
}


PVideoFrame __stdcall YV12To422::
GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);

    // check for crop left
    if (((uintptr_t)src->GetReadPtr(PLANAR_Y) |
         (uintptr_t)src->GetReadPtr(PLANAR_U) |
         (uintptr_t)src->GetReadPtr(PLANAR_V)) & (memalign - 1)) {
        int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        PVideoFrame alt = env->NewVideoFrame(vi_src, memalign);
        for (auto p : planes) {
            env->BitBlt(alt->GetWritePtr(p), alt->GetPitch(p),
                src->GetReadPtr(p), src->GetPitch(p),
                src->GetRowSize(p), src->GetHeight(p));
        }
        src = alt;
    }

    const int width_uv = aligned_size(src->GetRowSize(PLANAR_U), memalign);
    const int src_height_uv = src->GetHeight(PLANAR_U);
    const int src_pitch_y = src->GetPitch(PLANAR_Y);
    int src_pitch_u = src->GetPitch(PLANAR_U);
    int src_pitch_v = src_pitch_u;

    const uint8_t* srcpy = src->GetReadPtr(PLANAR_Y);
    const uint8_t* srcpu = src->GetReadPtr(PLANAR_U);
    const uint8_t* srcpv = src->GetReadPtr(PLANAR_V);

    int buff_pitch = aligned_size(src_pitch_u, memalign);
    uint8_t* buffu = nullptr;
    uint8_t* buffv = nullptr;
    if (lshift) {
        buffu = (uint8_t*)_mm_malloc(buff_pitch * src_height_uv * 2, memalign);
        buffv = buffu + buff_pitch * src_height_uv;
    }

    PVideoFrame yv16 = env->NewVideoFrame(vi_yv16, memalign);
    const int yv16_pitch_uv = yv16->GetPitch(PLANAR_U);
    uint8_t* yv16pu = yv16->GetWritePtr(PLANAR_U);
    uint8_t* yv16pv = yv16->GetWritePtr(PLANAR_V);

    omp_set_num_threads(num_threads);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (lshift) {
                proc_chroma_qpel_shift_h(width_uv, src_height_uv, srcpu, buffu,
                                         src_pitch_u, buff_pitch);
                srcpu = buffu;
                src_pitch_u = buff_pitch;
            }
            proc_chroma(width_uv, src_height_uv, srcpu, yv16pu, src_pitch_u,
                        yv16_pitch_uv, cubic_coefficients);
        }

        #pragma omp section
        {
            if (lshift) {
                proc_chroma_qpel_shift_h(width_uv, src_height_uv, srcpv, buffv,
                                         src_pitch_v, buff_pitch);
                srcpv = buffv;
                src_pitch_v = buff_pitch;
            }
            proc_chroma(width_uv, src_height_uv, srcpv, yv16pv,
                        src_pitch_v * dvpal, yv16_pitch_uv * dvpal, cubic_coefficients);
        }
    }

    if (lshift) {
        _mm_free((void*)buffu);
    }

    if (!yuy2out) {
        env->BitBlt(yv16->GetWritePtr(PLANAR_Y), yv16->GetPitch(PLANAR_Y),
                    srcpy, src_pitch_y, vi.width, vi.height);
        return yv16;
    }

    PVideoFrame dst = env->NewVideoFrame(vi_yuy2, memalign);

    yv16toyuy2(yv16->GetRowSize(PLANAR_U), vi.height, srcpy, yv16pu, yv16pv,
               dst->GetWritePtr(), dst->GetRowSize(), src_pitch_y,
               yv16_pitch_uv, dst->GetPitch());

    return dst;
}

extern int has_avx2();

static AVSValue __cdecl
create_yv12to422(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const VideoInfo& vi = clip->GetVideoInfo();
    if (!vi.IsYV12()) {
        env->ThrowError("YV12To422: input must be YV12.\n");
    }

    if (vi.height % 4 > 0) {
        env->ThrowError("YV12To422: height must be mod 4.\n");
    }
    if (vi.height < 16) {
        env->ThrowError(
            "YV12To422: height must be 16 or more.");
    }

    bool interlaced = args[1].AsBool(false);

    int itype = args[2].AsInt(2);
    if (itype < 0 || itype > 2) {
        env->ThrowError("YV12To422: itype must be set to 0, 1, or 2.\n");
    }

    int cplace = args[3].AsInt(interlaced ? 2 : 1);
    if (cplace < 0 || cplace > 3) {
        env->ThrowError("YV12To422: cplace must be set to 0, 1, 2, or 3.");
    }

    bool avx2 = false;
    if (args[7].AsBool(false) && has_avx2()) {
        avx2 = true;
    }

    return new YV12To422(clip, itype, interlaced, cplace, args[8].AsFloat(0.0),
                         args[9].AsFloat(0.75), args[5].AsBool(true), avx2,
                         args[4].AsBool(false), args[6].AsBool(false) ? 2 : 1,
                         env);
}


static const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("YV12To422",
                     /* 0*/ "c"
                     /* 1*/ "[interlaced]b"
                     /* 2*/ "[itype]i"
                     /* 3*/ "[cplace]i"
                     /* 4*/ "[lshift]b"
                     /* 5*/ "[yuy2]b"
                     /* 6*/ "[threads]b"
                     /* 7*/ "[avx2]b"
                     /* 8*/ "[b]f"
                     /* 9*/ "[c]f",

                     create_yv12to422, nullptr);
    return "YV12To422 ver." YV12TO422_VERSION " by OKA Motofumi";
}
