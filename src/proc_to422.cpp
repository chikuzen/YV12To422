/*
  proc_to422.cpp

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


#include <cstdint>
#include <tuple>
#include <map>

#include "proc_to422.h"
#include "simd.h"



///////////////// itype 0 (Point) /////////////////

template <typename T>
static void __stdcall
proc_point_p(const int width, const int height, const uint8_t* srcp,
             uint8_t* dstp, int src_pitch, int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < w; ++x) {
            T reg = load_reg(s + x);
            stream_reg(d + x, reg);
            stream_reg(d + dp + x, reg);
        }
        s += sp;
        d += 2 * dp;
    }
}


template <typename T>
static void __stdcall
proc_point_i(const int width, const int height, const uint8_t* srcp,
             uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    if (sp < 0) { // cplace=3(DV-PAL) and V-plane
        s += -sp * (height - 1);
        d += -dp * (2 * height - 1);
    }

    for (int y = 0; y < height; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg = load_reg(s + x);
            stream_reg(d + x, reg);
            stream_reg(d + 2 * dp + x, reg);
            reg = load_reg(s + sp + x);
            stream_reg(d + dp + x, reg);
            stream_reg(d + dp * 3 + x, reg);
        }
        s += 2 * sp;
        d += 4 * dp;
    }
}



///////////////// itype 1 (Linear) /////////////////

template <typename T>
static void __stdcall
proc_linear_c0_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < w; ++x) {
            T reg0 = load_reg(s + x);
            T reg1 = load_reg(s + sp + x);
            T avg = average(reg0, reg1);
            stream_reg(d + x, reg0);
            stream_reg(d + dp + x, avg);
        }
        s += sp;
        d += 2 * dp;
    }
    for (int x = 0; x < w; ++x) {
        T reg = load_reg(s + x);
        stream_reg(d + x, reg);
        stream_reg(d + dp + x, reg);
    }
}


template <typename T>
static void __stdcall
proc_linear_c0_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    if (sp < 0) { // cplace=3(DV-PAL) and V-plane
        s += -sp * (height - 1);
        d += -dp * (2 * height - 1);
    }

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg0 = load_reg(s + 0 * sp + x);
            T reg1 = load_reg(s + 1 * sp + x);
            T reg2 = load_reg(s + 2 * sp + x);
            T reg3 = load_reg(s + 3 * sp + x);

            stream_reg(d + 0 * dp + x, reg0);
            stream_reg(d + 1 * dp + x, reg1);
            stream_reg(d + 2 * dp + x, average(reg0, reg2));
            stream_reg(d + 3 * dp + x, average(reg1, reg3));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    for (int x = 0; x < w; ++x) {
        T reg0 = load_reg(s      + x);
        T reg1 = load_reg(s + sp + x);
        stream_reg(d          + x, reg0);
        stream_reg(d + 1 * dp + x, reg1);
        stream_reg(d + 2 * dp + x, reg0);
        stream_reg(d + 3 * dp + x, reg1);
    }
}


template <typename T>
static void __stdcall
proc_linear_c1_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    memcpy(d, s, width);
    d += dp;

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < w; ++x) {
            T reg0 = load_reg(s + x);
            T reg1 = load_reg(s + sp + x);
            stream_reg(d +      x, average(reg0, reg0, reg0, reg1));
            stream_reg(d + dp + x, average(reg0, reg1, reg1, reg1));
        }
        s += sp;
        d += 2 * dp;
    }

    memcpy(d, s, width);
}


template <typename T>
static void __stdcall
proc_linear_c1_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    memcpy(d, s, width);
    memcpy(d + dp, s + sp, width);
    d += 2 * dp;

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg0 = load_reg(s + 0 * sp + x);
            T reg1 = load_reg(s + 1 * sp + x);
            T reg2 = load_reg(s + 2 * sp + x);
            T reg3 = load_reg(s + 3 * sp + x);
            stream_reg(d + 0 * dp + x, average(reg0, reg0, reg0, reg2));
            stream_reg(d + 1 * dp + x, average(reg1, reg1, reg1, reg3));
            stream_reg(d + 2 * dp + x, average(reg0, reg2, reg2, reg2));
            stream_reg(d + 3 * dp + x, average(reg1, reg3, reg3, reg3));
        }
        s += 2 * sp;
        d += 4 * dp;
    }

    memcpy(d, s, width);
    memcpy(d + dp, s + sp, width);
}


template <typename T>
static void __stdcall
proc_linear_c2_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    T v85, v128, v171;
    set1_epi16(v85, 85);
    set1_epi16(v128, 128);
    set1_epi16(v171, 171);

    for (int y = 0; y < height - 2; y += 2) {
        memcpy(d, s, width);
        s += sp;
        d += dp;
        memcpy(d, s, width);
        d += dp;
        for (int x = 0; x < w; ++x) {
            T reg0, reg1, reg2, reg3, t0, t1;
            reg0 = load_reg(s + 0 * sp + x);
            reg2 = load_reg(s + 1 * sp + x);
            cvtepu8_epi16x2(reg0, reg1);
            cvtepu8_epi16x2(reg2, reg3);

            t0 = add_epu16(mullo_epi16(reg0, v171), mullo_epi16(reg2, v85));
            t0 = srli_epi16(add_epu16(t0, v128), 8);
            t1 = add_epu16(mullo_epi16(reg1, v171), mullo_epi16(reg3, v85));
            t1 = srli_epi16(add_epu16(t1, v128), 8);
            stream_reg(d + x, packus_epi16(t0, t1));

            t0 = add_epu16(mullo_epi16(reg0, v85), mullo_epi16(reg2, v171));
            t0 = srli_epi16(add_epu16(t0, v128), 8);
            t1 = add_epu16(mullo_epi16(reg1, v85), mullo_epi16(reg3, v171));
            t1 = srli_epi16(add_epu16(t1, v128), 8);
            stream_reg(d + dp + x, packus_epi16(t0, t1));
        }
        s += sp;
        d += 2 * dp;
    }
    memcpy(d, s, width);
    s += sp;
    d += dp;
    memcpy(d, s, width);
    memcpy(d + dp, s, width);
    memcpy(d + 2 * dp, s, width);
}


template <typename T>
static void __stdcall
proc_linear_c2_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    memcpy(d, s, width);
    d += dp;
    memcpy(d, s + sp, width);
    d += dp;

    T v4;
    set1_epi16(v4, 4);

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg0, reg1, reg2, reg3, t0, t1;

            reg0 = load_reg(s + 0 * sp + x);
            reg2 = load_reg(s + 2 * sp + x);
            cvtepu8_epi16x2(reg0, reg1);
            cvtepu8_epi16x2(reg2, reg3);
            
            t0 = add_epu16(slli_epi16(reg0, 2), reg0); //t0=5*reg0
            t0 = add_epu16(t0, add_epu16(slli_epi16(reg2, 1), reg2));//t0=t0+3*reg2
            t0 = srli_epi16(add_epu16(t0, v4), 3); //t0=(t0+4)>>3
            t1 = add_epu16(slli_epi16(reg1, 2), reg1);
            t1 = add_epu16(t1, add_epu16(slli_epi16(reg3, 1), reg3));
            t1 = srli_epi16(add_epu16(t1, v4), 3);
            stream_reg(d + 0 * dp + x, packus_epi16(t0, t1));

            t0 = add_epu16(reg0, sub_epi16(slli_epi16(reg2, 3), reg2)); //t0=reg0+7*reg2
            t0 = srli_epi16(add_epu16(t0, v4), 3); //t0=(t0+4)>>3
            t1 = add_epu16(reg1, sub_epi16(slli_epi16(reg3, 3), reg3));
            t1 = srli_epi16(add_epu16(t1, v4), 3);
            stream_reg(d + 2 * dp + x, packus_epi16(t0, t1));

            reg0 = load_reg(s + 1 * sp + x);
            reg2 = load_reg(s + 3 * sp + x);
            cvtepu8_epi16x2(reg0, reg1);
            cvtepu8_epi16x2(reg2, reg3);

            t0 = add_epu16(sub_epi16(slli_epi16(reg0, 3), reg0), reg2); //t0=7*reg0-reg2
            t0 = srli_epi16(add_epu16(t0, v4), 3); //t0=(t0+4)>>3
            t1 = add_epu16(sub_epi16(slli_epi16(reg1, 3), reg1), reg3);
            t1 = srli_epi16(add_epu16(t1, v4), 3);
            stream_reg(d + 1 * dp + x, packus_epi16(t0, t1));

            t0 = add_epu16(slli_epi16(reg0, 1), reg0); //t0=3*reg0
            t0 = add_epu16(t0, add_epu16(slli_epi16(reg2, 2), reg2)); //t0=t0+5*reg2
            t0 = srli_epi16(add_epu16(t0, v4), 3); //t0=(t0+4)>>3
            t1 = add_epu16(slli_epi16(reg1, 1), reg1);
            t1 = add_epu16(t1, add_epu16(slli_epi16(reg3, 2), reg3));
            t1 = srli_epi16(add_epu16(t1, v4), 3);
            stream_reg(d + 3 * dp + x, packus_epi16(t0, t1));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    memcpy(d, s, width);
    memcpy(d + dp, s + sp, width);
}

/*
//this is a bit faster, but inaccurate.
template <typename T>
static void __stdcall
proc_linear_c2_i_f(const int width, const int height, const uint8_t* srcp,
uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    memcpy(d, s, width);
    d += dp;
    memcpy(d, s + sp, width);
    d += dp;

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg0, reg1;

            reg0 = load_reg(s + 0 * sp + x);
            reg1 = load_reg(s + 2 * sp + x);

            stream_reg(d + 0 * dp + x, average(reg0, average(reg0, reg1, reg1, reg1)));
            stream_reg(d + 2 * dp + x, average(average(reg0, reg1, reg1, reg1), reg1));

            reg0 = load_reg(s + 1 * sp + x);
            reg1 = load_reg(s + 3 * sp + x);

            stream_reg(d + 1 * dp + x, average(reg0, average(reg0, reg0, reg0, reg1)));
            stream_reg(d + 3 * dp + x, average(average(reg0, reg0, reg0, reg1), reg1));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    memcpy(d, s, width);
    memcpy(d + dp, s + sp, width);
}
*/

template <typename T>
static void __stdcall
proc_linear_c3_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    T v85, v128, v171;
    set1_epi16(v85, 85);
    set1_epi16(v128, 128);
    set1_epi16(v171, 171);

    memcpy(d, s, width);
    memcpy(d + dp, s, width);
    s += sp;
    d += 2 * dp;

    for (int y = 1; y < height - 1; y += 2) {
        for (int x = 0; x < w; ++x) {
            T reg0, reg1, reg2, reg3, t0, t1;
            reg0 = load_reg(s      + x);
            reg2 = load_reg(s + sp + x);
            stream_reg(d + 0 * dp + x, reg0);
            stream_reg(d + 3 * dp + x, reg2);

            cvtepu8_epi16x2(reg0, reg1);
            cvtepu8_epi16x2(reg2, reg3);

            t0 = add_epu16(mullo_epi16(reg0, v171), mullo_epi16(reg2, v85));
            t0 = srli_epi16(add_epu16(t0, v128), 8);
            t1 = add_epu16(mullo_epi16(reg1, v171), mullo_epi16(reg3, v85));
            t1 = srli_epi16(add_epu16(t1, v128), 8);
            stream_reg(d + 1 * dp + x, packus_epi16(t0, t1));

            t0 = add_epu16(mullo_epi16(reg0, v85), mullo_epi16(reg2, v171));
            t0 = srli_epi16(add_epu16(t0, v128), 8);
            t1 = add_epu16(mullo_epi16(reg1, v85), mullo_epi16(reg3, v171));
            t1 = srli_epi16(add_epu16(t1, v128), 8);
            stream_reg(d + 2 * dp + x, packus_epi16(t0, t1));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    memcpy(d, s, width);
    memcpy(d + dp, s, width);
}


//////////////// itype 2 (cubic) /////////////////////////

// not implemented yet


//////////////////////////////////////////////////////////////////////////////

proc_to422 get_proc_chroma(int itype, int cplace, bool interlaced, bool avx2)
{
    //      <itype, cplace, interlaced, avx2>
    std::map<std::tuple<int, int, bool, bool>, proc_to422> func;

    func[std::make_tuple(0, 0, false, false)] = proc_point_p<__m128i>;
    func[std::make_tuple(0, 0, false, true)]  = proc_point_p<__m256i>;
    func[std::make_tuple(0, 0, true,  false)] = proc_point_i<__m128i>;
    func[std::make_tuple(0, 0, true,  true)]  = proc_point_i<__m256i>;
    func[std::make_tuple(0, 1, false, false)] = proc_point_p<__m128i>;
    func[std::make_tuple(0, 1, false, true)]  = proc_point_p<__m256i>;
    func[std::make_tuple(0, 1, true,  false)] = proc_point_i<__m128i>;
    func[std::make_tuple(0, 1, true,  true)]  = proc_point_i<__m256i>;
    func[std::make_tuple(0, 2, false, false)] = proc_point_p<__m128i>;
    func[std::make_tuple(0, 2, false, true)]  = proc_point_p<__m256i>;
    func[std::make_tuple(0, 2, true,  false)] = proc_point_i<__m128i>;
    func[std::make_tuple(0, 2, true,  true)]  = proc_point_i<__m256i>;
    func[std::make_tuple(0, 3, false, false)] = proc_point_p<__m128i>;
    func[std::make_tuple(0, 3, false, true)]  = proc_point_p<__m256i>;
    func[std::make_tuple(0, 3, true,  false)] = proc_point_i<__m128i>;
    func[std::make_tuple(0, 3, true,  true)]  = proc_point_i<__m256i>;
    func[std::make_tuple(1, 0, false, false)] = proc_linear_c0_p<__m128i>;
    func[std::make_tuple(1, 0, false, true)]  = proc_linear_c0_p<__m256i>;
    func[std::make_tuple(1, 0, true,  false)] = proc_linear_c0_i<__m128i>;
    func[std::make_tuple(1, 0, true,  true)]  = proc_linear_c0_i<__m256i>;
    func[std::make_tuple(1, 1, false, false)] = proc_linear_c1_p<__m128i>;
    func[std::make_tuple(1, 1, false, true)]  = proc_linear_c1_p<__m256i>;
    func[std::make_tuple(1, 1, true,  false)] = proc_linear_c1_i<__m128i>;
    func[std::make_tuple(1, 1, true,  true)]  = proc_linear_c1_i<__m256i>;
    func[std::make_tuple(1, 2, false, false)] = proc_linear_c2_p<__m128i>;
    func[std::make_tuple(1, 2, false, true)]  = proc_linear_c2_p<__m256i>;
    func[std::make_tuple(1, 2, true,  false)] = proc_linear_c2_i<__m128i>;
    func[std::make_tuple(1, 2, true,  true)]  = proc_linear_c2_i<__m256i>;
    func[std::make_tuple(1, 3, false, false)] = proc_linear_c3_p<__m128i>;
    func[std::make_tuple(1, 3, false, true)]  = proc_linear_c3_p<__m256i>;
    func[std::make_tuple(1, 3, true,  false)] = proc_linear_c0_i<__m128i>;
    func[std::make_tuple(1, 3, true,  true)]  = proc_linear_c0_i<__m256i>;

    return func[std::make_tuple(itype, cplace, interlaced, avx2)];
}

