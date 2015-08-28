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
             uint8_t* dstp, int src_pitch, int dst_pitch,
             const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < w; ++x) {
            T reg = load_reg(s + x);
            stream_reg(d      + x, reg);
            stream_reg(d + dp + x, reg);
        }
        s += sp;
        d += 2 * dp;
    }
}


template <typename T>
static void __stdcall
proc_point_i(const int width, const int height, const uint8_t* srcp,
             uint8_t* dstp, const int src_pitch, const int dst_pitch,
             const int16_t* coeffs)
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
            stream_reg(d + 0 * dp + x, reg);
            stream_reg(d + 2 * dp + x, reg);
            reg = load_reg(s + sp + x);
            stream_reg(d + 1 * dp + x, reg);
            stream_reg(d + 3 * dp + x, reg);
        }
        s += 2 * sp;
        d += 4 * dp;
    }
}



///////////////// itype 1 (Linear) /////////////////

template <typename T>
static void __stdcall
proc_linear_c0_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < w; ++x) {
            T reg0 = load_reg(s      + x);
            T reg1 = load_reg(s + sp + x);
            T avg = average(reg0, reg1);
            stream_reg(d      + x, reg0);
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
proc_linear_c03_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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
        stream_reg(d + 0 * dp + x, reg0);
        stream_reg(d + 1 * dp + x, reg1);
        stream_reg(d + 2 * dp + x, reg0);
        stream_reg(d + 3 * dp + x, reg1);
    }
}


template <typename T>
static void __stdcall
proc_linear_c1_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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
            T reg0 = load_reg(s      + x);
            T reg1 = load_reg(s + sp + x);
            stream_reg(d      + x, average(reg0, reg0, reg0, reg1));
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
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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
    memcpy(d + 0 * dp, s, width);
    memcpy(d + 1 * dp, s, width);
    memcpy(d + 2 * dp, s, width);
}


template <typename T>
static void __stdcall
proc_linear_c2_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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


template <typename T>
static void __stdcall
proc_linear_c3_p(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
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

template <typename T>
static __forceinline T
cubic(const T& a, const T& b, const T& c, const T& d, const T& coef0, const T& coef1)
{
    T zero, total0, total1, total2, total3;
    zero = xor_reg(a, a);
    set1_epi32(total0, 512);
    total1 = total0;
    total2 = total0;
    total3 = total0;

    T ab_lo = unpacklo_epi8(a, b);
    T ab_hi = unpackhi_epi8(a, b);

    T ab0 = unpacklo_epi8(ab_lo, zero);
    T ab1 = unpackhi_epi8(ab_lo, zero);
    T ab2 = unpacklo_epi8(ab_hi, zero);
    T ab3 = unpackhi_epi8(ab_hi, zero);
    total0 = add_epi32(madd_epi16(ab0, coef0), total0);
    total1 = add_epi32(madd_epi16(ab1, coef0), total1);
    total2 = add_epi32(madd_epi16(ab2, coef0), total2);
    total3 = add_epi32(madd_epi16(ab3, coef0), total3);

    T cd_lo = unpacklo_epi8(c, d);
    T cd_hi = unpackhi_epi8(c, d);

    T cd0 = unpacklo_epi8(cd_lo, zero);
    T cd1 = unpackhi_epi8(cd_lo, zero);
    T cd2 = unpacklo_epi8(cd_hi, zero);
    T cd3 = unpackhi_epi8(cd_hi, zero);
    total0 = add_epi32(madd_epi16(cd0, coef1), total0);
    total1 = add_epi32(madd_epi16(cd1, coef1), total1);
    total2 = add_epi32(madd_epi16(cd2, coef1), total2);
    total3 = add_epi32(madd_epi16(cd3, coef1), total3);

    total0 = srli_epi32(total0, 10);
    total1 = srli_epi32(total1, 10);
    total2 = srli_epi32(total2, 10);
    total3 = srli_epi32(total3, 10);

    total0 = packs_epi32(total0, total1);
    total1 = packs_epi32(total2, total3);
    return packus_epi16(total0, total1);
}


template <typename T>
static __forceinline T
cubic_flip(const T& a, const T& b, const T& c, const T& d, const T& coef0, const T& coef1)
{
    return cubic(d, c, b, a, coef0, coef1);
}

template <typename T>
static __forceinline T
cubic_symmetry(const T& a, const T& b, const T& c, const T& d, const T& coeff)
{
    T zero, total0, total1, total2, total3;
    zero = xor_reg(a, a);
    set1_epi32(total0, 512);
    total1 = total0;
    total2 = total0;
    total3 = total0;

    T ad_lo = add_epu16(unpacklo_epi8(a, zero), unpacklo_epi8(d, zero));
    T ad_hi = add_epu16(unpackhi_epi8(a, zero), unpackhi_epi8(d, zero));

    T bc_lo = add_epu16(unpacklo_epi8(b, zero), unpacklo_epi8(c, zero));
    T bc_hi = add_epu16(unpackhi_epi8(b, zero), unpackhi_epi8(c, zero));

    T adbc0 = unpacklo_epi16(ad_lo, bc_lo);
    T adbc1 = unpackhi_epi16(ad_lo, bc_lo);
    T adbc2 = unpacklo_epi16(ad_hi, bc_hi);
    T adbc3 = unpackhi_epi16(ad_hi, bc_hi);

    total0 = add_epi32(madd_epi16(adbc0, coeff), total0);
    total1 = add_epi32(madd_epi16(adbc1, coeff), total1);
    total2 = add_epi32(madd_epi16(adbc2, coeff), total2);
    total3 = add_epi32(madd_epi16(adbc3, coeff), total3);

    total0 = srli_epi32(total0, 10);
    total1 = srli_epi32(total1, 10);
    total2 = srli_epi32(total2, 10);
    total3 = srli_epi32(total3, 10);

    total0 = packs_epi32(total0, total1);
    total1 = packs_epi32(total2, total3);
    return packus_epi16(total0, total1);

}

template <typename T>
static void __stdcall
proc_cubic_c0_p(const int width, const int height, const uint8_t* srcp,
                uint8_t* dstp, const int src_pitch, const int dst_pitch,
                const int16_t* coeffs)
{
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / sizeof(T);
    const int dp = dst_pitch / sizeof(T);

    const T* s0 = (T*)srcp + 2 * sp;
    const T* s1 = (T*)srcp;
    const T* s2 = s1 + sp;
    const T* s3 = s2 + sp;

    T coeff;
    set1_epi32(coeff, ((int32_t*)coeffs)[0]);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < w; ++x) {
            T src0 = load_reg(s0 + x);
            T src1 = load_reg(s1 + x);
            T src2 = load_reg(s2 + x);
            T src3 = load_reg(s3 + x);

            stream_reg(d + x, src1);

            T cs = cubic_symmetry(src0, src1, src2, src3, coeff);
            stream_reg(d + dp + x, cs);
        }
        s0 = s1;
        s1 = s2;
        s2 = s2 + (y < height - 2 ? sp : 0);
        s3 = y < height - 3 ? s3 + sp : s0;
        d += 2 * dp;
    }
}


template <typename T>
static void __stdcall
proc_cubic_c03_i(const int width, const int height, const uint8_t* srcp,
                 uint8_t* dstp, const int src_pitch, const int dst_pitch,
                 const int16_t* coeffs)
{
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    const T* s1 = (T*)srcp;
    if (sp < 0) { // cplace=3(DV-PAL) and V-plane
        s1 += -sp * (height - 1);
        d += -dp * (2 * height - 1);
    }
    const T* s0 = s1 + 4 * sp;
    const T* s2 = s1 + 2 * sp;
    const T* s3 = s1 + 4 * sp;

    T coeff;
    set1_epi32(coeff, ((int32_t*)coeffs)[0]);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < w; ++x) {
            T src0 = load_reg(s0 + x);
            T src1 = load_reg(s1 + x);
            T src2 = load_reg(s2 + x);
            T src3 = load_reg(s3 + x);

            stream_reg(d + x, src1);

            T cs = cubic_symmetry(src0, src1, src2, src3, coeff);
            stream_reg(d + 2 * dp + x, cs);
        }
        s1 += sp;
        s0 = s1 + (y < 2 ? 4 : - 2) * sp;
        s2 = s1 + (y > height - 2 ? 0 : 2) * sp;
        s3 = s1 + (y > height - 4 ? -2 : 4) * sp;
        d += (y & 1 ? 3 : 1) * dp;
    }
}

template <typename T>
static void __stdcall
proc_cubic_c1_p(const int width, const int height, const uint8_t* srcp,
    uint8_t* dstp, const int src_pitch, const int dst_pitch,
    const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    T coeff0, coeff1, src0, src1, src2, src3;
    set1_epi32(coeff0, ((int32_t*)coeffs)[0]);
    set1_epi32(coeff1, ((int32_t*)coeffs)[1]);

    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + 2 * sp + x);
        src1 = load_reg(s + 0 * sp + x);
        src2 = load_reg(s + 1 * sp + x);
        stream_reg(d + 0 * dp + x, cubic(src2, src1, src1, src2, coeff0, coeff1));
        stream_reg(d + 1 * dp + x, cubic_flip(src0, src1, src2, src0, coeff0, coeff1));
        stream_reg(d + 2 * dp + x, cubic(src0, src1, src2, src0, coeff0, coeff1));
    }
    d += 3 * dp;

    for (int y = 0; y < height - 3; ++y) {
        for (int x = 0; x < w; ++x) {
            src0 = load_reg(s + 0 * sp + x);
            src1 = load_reg(s + 1 * sp + x);
            src2 = load_reg(s + 2 * sp + x);
            src3 = load_reg(s + 3 * sp + x);

            stream_reg(d + x, cubic_flip(src0, src1, src2, src3, coeff0, coeff1));
            stream_reg(d + dp + x, cubic(src0, src1, src2, src3, coeff0, coeff1));
        }
        s += sp;
        d += 2 * dp;
    }

    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + 0 * sp + x);
        src1 = load_reg(s + 1 * sp + x);
        src2 = load_reg(s + 2 * sp + x);
        stream_reg(d + 0 * dp + x, cubic_flip(src0, src1, src2, src0, coeff0, coeff1));
        stream_reg(d + 1 * dp + x, cubic(src0, src1, src2, src0, coeff0, coeff1));
        stream_reg(d + 2 * dp + x, cubic_flip(src1, src2, src2, src1, coeff0, coeff1));
    }
}


template <typename T>
static void __stdcall
proc_cubic_c12_i(const int width, const int height, const uint8_t* srcp,
    uint8_t* dstp, const int src_pitch, const int dst_pitch,
    const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    T coeff0, coeff1, coeff2, coeff3, src0, src1, src2, src3;
    set1_epi32(coeff0, ((int32_t*)coeffs)[0]);
    set1_epi32(coeff1, ((int32_t*)coeffs)[1]);
    set1_epi32(coeff2, ((int32_t*)coeffs)[2]);
    set1_epi32(coeff3, ((int32_t*)coeffs)[3]);

    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + 0 * sp + x);
        src1 = load_reg(s + 2 * sp + x);
        src2 = load_reg(s + 4 * sp + x);
        stream_reg(d + 0 * dp + x, cubic(src1, src0, src0, src1, coeff0, coeff1));
        stream_reg(d + 2 * dp + x, cubic_flip(src2, src0, src1, src2, coeff2, coeff3));
        stream_reg(d + 4 * dp + x, cubic(src2, src0, src1, src2, coeff0, coeff1));
        src0 = load_reg(s + 1 * sp + x);
        src1 = load_reg(s + 3 * sp + x);
        src2 = load_reg(s + 5 * sp + x);
        stream_reg(d + 1 * dp + x, cubic(src1, src0, src0, src1, coeff2, coeff3));
        stream_reg(d + 3 * dp + x, cubic_flip(src2, src0, src1, src2, coeff0, coeff1));
        stream_reg(d + 5 * dp + x, cubic(src2, src0, src1, src2, coeff2, coeff3));
    }
    d += 6 * dp;

    for (int y = 0; y < height - 6; y += 2) {
        for (int x = 0; x < w; ++x) {
            src0 = load_reg(s + 0 * sp + x);
            src1 = load_reg(s + 2 * sp + x);
            src2 = load_reg(s + 4 * sp + x);
            src3 = load_reg(s + 6 * sp + x);
            stream_reg(d + 0 * dp + x, cubic_flip(src0, src1, src2, src3, coeff2, coeff3));
            stream_reg(d + 2 * dp + x, cubic(src0, src1, src2, src3, coeff0, coeff1));
        }
        for (int x = 0; x < w; ++x) {
            src0 = load_reg(s + 1 * sp + x);
            src1 = load_reg(s + 3 * sp + x);
            src2 = load_reg(s + 5 * sp + x);
            src3 = load_reg(s + 7 * sp + x);
            stream_reg(d + 1 * dp + x, cubic_flip(src0, src1, src2, src3, coeff0, coeff1));
            stream_reg(d + 3 * dp + x, cubic(src0, src1, src2, src3, coeff2, coeff3));
        }
        s += 2 * sp;
        d += 4 * dp;
    }

    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + 0 * sp + x);
        src1 = load_reg(s + 2 * sp + x);
        src2 = load_reg(s + 4 * sp + x);
        stream_reg(d + 0 * dp + x, cubic_flip(src0, src1, src2, src0, coeff2, coeff3));
        stream_reg(d + 2 * dp + x, cubic(src0, src1, src2, src0, coeff0, coeff1));
        stream_reg(d + 4 * dp + x, cubic_flip(src1, src2, src2, src1, coeff2, coeff3));
        src0 = load_reg(s + 1 * sp + x);
        src1 = load_reg(s + 3 * sp + x);
        src2 = load_reg(s + 5 * sp + x);
        stream_reg(d + 1 * dp + x, cubic_flip(src0, src1, src2, src0, coeff0, coeff1));
        stream_reg(d + 3 * dp + x, cubic(src0, src1, src2, src0, coeff2, coeff3));
        stream_reg(d + 5 * dp + x, cubic_flip(src1, src2, src2, src1, coeff0, coeff1));
    }
}

template <typename T>
static void __stdcall
proc_cubic_c2_p(const int width, const int height, const uint8_t* srcp,
    uint8_t* dstp, const int src_pitch, const int dst_pitch,
    const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    T coeff0, coeff1, src0, src1, src2, src3;
    set1_epi32(coeff0, ((int32_t*)coeffs)[0]);
    set1_epi32(coeff1, ((int32_t*)coeffs)[1]);

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            src0 = load_reg(s + 0 * sp + x);
            src1 = load_reg(s + 1 * sp + x);
            src2 = load_reg(s + 2 * sp + x);
            src3 = load_reg(s + 3 * sp + x);

            stream_reg(d + 0 * dp + x, src0);
            stream_reg(d + 1 * dp + x, src1);
            stream_reg(d + 2 * dp + x, cubic(src0, src1, src2, src3, coeff0, coeff1));
            stream_reg(d + 3 * dp + x, cubic_flip(src0, src1, src2, src3, coeff0, coeff1));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + x);
        src1 = load_reg(s + sp + x);
        stream_reg(d + 0 * dp + x, src0);
        stream_reg(d + 1 * dp + x, src1);
        stream_reg(d + 2 * dp + x, cubic(src0, src1, src1, src0, coeff0, coeff1));
        stream_reg(d + 3 * dp + x, cubic_flip(src0, src1, src1, src0, coeff0, coeff1));
    }
}


template <typename T>
static void __stdcall
proc_cubic_c3_p(const int width, const int height, const uint8_t* srcp,
    uint8_t* dstp, const int src_pitch, const int dst_pitch,
    const int16_t* coeffs)
{
    const T* s = (T*)srcp;
    T* d = (T*)dstp;
    const int w = width / sizeof(T);
    const int sp = src_pitch / (int)sizeof(T);
    const int dp = dst_pitch / (int)sizeof(T);

    T coeff0, coeff1, src0, src1, src2, src3;
    set1_epi32(coeff0, ((int32_t*)coeffs)[0]);
    set1_epi32(coeff1, ((int32_t*)coeffs)[1]);

    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + x);
        src1 = load_reg(s + sp + x);
        stream_reg(d + x, cubic(src1, src0, src0, src1, coeff0, coeff1));
    }
    d += dp;

    for (int y = 0; y < height - 2; y += 2) {
        for (int x = 0; x < w; ++x) {
            src0 = load_reg(s + 0 * sp + x);
            src1 = load_reg(s + 1 * sp + x);
            src2 = load_reg(s + 2 * sp + x);
            src3 = load_reg(s + 3 * sp + x);
            stream_reg(d + 0 * dp + x, src0);
            stream_reg(d + 1 * dp + x, src1);
            stream_reg(d + 2 * dp + x, cubic_flip(src0, src1, src2, src3, coeff0, coeff1));
            stream_reg(d + 3 * dp + x, cubic(src0, src1, src2, src3, coeff0, coeff1));
        }
        s += 2 * sp;
        d += 4 * dp;
    }
    for (int x = 0; x < w; ++x) {
        src0 = load_reg(s + x);
        src1 = load_reg(s + sp + x);
        stream_reg(d + 0 * dp + x, src0);
        stream_reg(d + 1 * dp + x, src1);
        stream_reg(d + 2 * dp + x, cubic_flip(src0, src1, src1, src0, coeff0, coeff1));
    }
}


/////////////////////////////////////////////////////////////////////////////


template <typename T>
static void __stdcall
proc_qpel_shift_h(const int width, const int height, const uint8_t* srcp,
                  uint8_t* dstp, const int src_pitch, const int dst_pitch)
{
    for (int y = 0; y < height; ++y) {

        T current = load_reg((T*)srcp);
        T left = slli_reg<1>(current);
        T mask = slli_reg<1>(cmpeq(current, current));
        left = blendv_epi8(current, left, mask);
        stream_reg((T*)dstp, average(current, current, current, left));

        for (int x = sizeof(T); x < width; x += sizeof(T)) {
            current = load_reg((T*)(srcp + x));
            left = loadu_reg((T*)(srcp + x - 1));
            stream_reg((T*)(dstp + x), average(current, current, current, left));
        }
        srcp += src_pitch;
        dstp += dst_pitch;
    }
}


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
    func[std::make_tuple(1, 0, true,  false)] = proc_linear_c03_i<__m128i>;
    func[std::make_tuple(1, 0, true,  true)]  = proc_linear_c03_i<__m256i>;
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
    func[std::make_tuple(1, 3, true,  false)] = proc_linear_c03_i<__m128i>;
    func[std::make_tuple(1, 3, true,  true)]  = proc_linear_c03_i<__m256i>;
    func[std::make_tuple(2, 0, false, false)] = proc_cubic_c0_p<__m128i>;
    func[std::make_tuple(2, 0, false, true)]  = proc_cubic_c0_p<__m256i>;
    func[std::make_tuple(2, 0, true,  false)] = proc_cubic_c03_i<__m128i>;
    func[std::make_tuple(2, 0, true,  true)]  = proc_cubic_c03_i<__m256i>;
    func[std::make_tuple(2, 1, false, false)] = proc_cubic_c1_p<__m128i>;
    func[std::make_tuple(2, 1, false, true)]  = proc_cubic_c1_p<__m256i>;
    func[std::make_tuple(2, 1, true,  false)] = proc_cubic_c12_i<__m128i>;
    func[std::make_tuple(2, 1, true,  true)]  = proc_cubic_c12_i<__m256i>;
    func[std::make_tuple(2, 2, false, false)] = proc_cubic_c2_p<__m128i>;
    func[std::make_tuple(2, 2, false, true)]  = proc_cubic_c2_p<__m256i>;
    func[std::make_tuple(2, 2, true,  false)] = proc_cubic_c12_i<__m128i>;
    func[std::make_tuple(2, 2, true,  true)]  = proc_cubic_c12_i<__m256i>;
    func[std::make_tuple(2, 3, false, false)] = proc_cubic_c3_p<__m128i>;
    func[std::make_tuple(2, 3, false, true)]  = proc_cubic_c3_p<__m256i>;
    func[std::make_tuple(2, 3, true,  false)] = proc_cubic_c03_i<__m128i>;
    func[std::make_tuple(2, 3, true,  true)]  = proc_cubic_c03_i<__m256i>;

    return func[std::make_tuple(itype, cplace, interlaced, avx2)];
}

proc_horizontal get_proc_horizontal_shift(bool use_avx2)
{
    return use_avx2 ?
        proc_qpel_shift_h<__m256i> :
        proc_qpel_shift_h<__m128i>;
}
