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
#include <malloc.h>
#include <immintrin.h>
#include <windows.h>
#include "avisynth.h"

#include "proc_to422.h"
#include "simd.h"


#define YV12TO422_VERSION "0.2.0"


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
            stream_reg((T*)(dstp + dpos), y0);

            dpos += sizeof(T);
            if (dst_rowsize < dpos) continue;
            stream_reg((T*)(dstp + dpos), y1);

            dpos += sizeof(T);
            if (dst_rowsize < dpos) continue;
            stream_reg((T*)(dstp + dpos), y2);

            dpos += sizeof(T);
            if (dst_rowsize < dpos) continue;
            stream_reg((T*)(dstp + dpos), y3);
        }
        srcpy += src_pitch_y;
        srcpu += src_pitch_uv;
        srcpv += src_pitch_uv;
        dstp += dst_pitch;
    }

}


class YV12To422 : public GenericVideoFilter
{
    VideoInfo vi_yv16;
    VideoInfo vi_src;
    bool yuy2out;
    bool lshift;
    int dvpal;
    int memalign;

    proc_to422 proc_chroma;
    proc_to422 proc_chroma_qpel_shift_h;
    planar_to_packed yv16toyuy2;


public:
    YV12To422(
        PClip child, int itype, bool interlaced, int cplace, double _b,
        double _c, bool yuy2, bool avx2, bool lshift, IScriptEnvironment* env);
    ~YV12To422() {};
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


YV12To422::
YV12To422(PClip _child, int itype, bool interlaced, int cplace, double b,
           double c, bool yuy2, bool avx2, bool _lshift,
           IScriptEnvironment* env)
  : GenericVideoFilter(_child),
    yuy2out(yuy2),
    lshift(_lshift)
{
    memalign = avx2 ? sizeof(__m256i) : sizeof(__m128i);
    proc_chroma = get_proc_chroma(itype, cplace, interlaced, avx2);
    proc_chroma_qpel_shift_h = get_proc_horizontal_shift(avx2);
    yv16toyuy2 = avx2 ? planar_to_yuy2<__m256i> : planar_to_yuy2<__m128i>;

    dvpal = interlaced && cplace == 3 ? -1 : 1;

    memcpy(&vi_src, &vi, sizeof(VideoInfo));
    memcpy(&vi_yv16, &vi, sizeof(VideoInfo));
    vi_yv16.pixel_type = VideoInfo::CS_YV16;
    vi.pixel_type = yuy2out ? VideoInfo::CS_YUY2 : VideoInfo::CS_YV16;

#ifdef DEBUG
    std::cerr << "cplace:" << cplace << " itype:" << itype << " interlaced:"
        << interlaced << " yuy2:" << yuy2out << " avx2:" << avx2 <<
        " threads: " << threads << "\n";
#endif
}


static inline int aligned_size(int x, int align)
{
    return ((x + align - 1) / align) * align;
}


PVideoFrame __stdcall YV12To422::
GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);

    // check for crop left
    if ((uintptr_t)src->GetReadPtr(PLANAR_Y) & (memalign - 1) ||
        (uintptr_t)src->GetReadPtr(PLANAR_U) & (memalign - 1) ||
        (uintptr_t)src->GetReadPtr(PLANAR_V) & (memalign - 1)) {
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
    int src_pitch_uv = src->GetPitch(PLANAR_U);

    const uint8_t* srcpy = src->GetReadPtr(PLANAR_Y);
    const uint8_t* srcpu = src->GetReadPtr(PLANAR_U);
    const uint8_t* srcpv = src->GetReadPtr(PLANAR_V);

    if (lshift) {
        int buff_pitch = aligned_size(src_pitch_uv, memalign);
        uint8_t* buffu =
            (uint8_t*)_mm_malloc(buff_pitch * src_height_uv * 2, memalign);
        uint8_t* buffv = buffu + buff_pitch * src_height_uv;

        proc_chroma_qpel_shift_h(width_uv, src_height_uv, srcpu, buffu,
                                 src_pitch_uv, buff_pitch);

        proc_chroma_qpel_shift_h(width_uv, src_height_uv, srcpv, buffv,
                                 src_pitch_uv, buff_pitch);

        srcpu = buffu;
        srcpv = buffv;
        src_pitch_uv = buff_pitch;
    }

    PVideoFrame yv16 = env->NewVideoFrame(vi_yv16, memalign);
    const int yv16_pitch_uv = yv16->GetPitch(PLANAR_U);
    uint8_t* yv16pu = yv16->GetWritePtr(PLANAR_U);
    uint8_t* yv16pv = yv16->GetWritePtr(PLANAR_V);

    proc_chroma(width_uv, src_height_uv, srcpu, yv16pu, src_pitch_uv,
                yv16_pitch_uv);

    proc_chroma(width_uv, src_height_uv, srcpv, yv16pv,
                src_pitch_uv * dvpal, yv16_pitch_uv * dvpal);

    if (lshift) {
        _mm_free((void*)srcpu);
    }

    if (!yuy2out) {
        env->BitBlt(yv16->GetWritePtr(PLANAR_Y), yv16->GetPitch(PLANAR_Y),
                    srcpy, src_pitch_y, vi.width, vi.height);
        return yv16;
    }

    PVideoFrame dst = env->NewVideoFrame(vi, memalign);

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

    int itype = args[1].AsInt(1);
    if (itype < 0 || itype > 1) {
        env->ThrowError("YV12To422: itype must be set to 0, or 1.\n");
    }

    bool interlaced = args[2].AsBool(false);
    int cplace = args[3].AsInt(interlaced ? 2 : 1);
    if (cplace < 0 || cplace > 3) {
        env->ThrowError("YV12To422: cplace must be set to 0, 1, 2, or 3.");
    }

    bool avx2 = false;
    if (args[6].AsBool(false) && has_avx2()) {
        avx2 = true;
    }

    return new YV12To422(clip, itype, interlaced, cplace, args[7].AsFloat(0.0),
                         args[8].AsFloat(0.75), args[5].AsBool(true), avx2,
                         args[4].AsBool(false), env);
}


static const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("YV12To422",
                     "c"
                     "[itype]i"
                     "[interlaced]b"
                     "[cplace]i"
                     "[lshift]b"
                     "[yuy2]b"
                     "[avx2]b"
                     "[b]f"
                     "[c]f",

                     create_yv12to422, nullptr);
    return "YV12To422 ver." YV12TO422_VERSION " by OKA Motofumi";
}
