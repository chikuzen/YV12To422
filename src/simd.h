/*
  simd.h

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


#ifndef YV12TO422_SIMD_H
#define YV12TO422_SIMD_H

#include <immintrin.h>


static __forceinline __m128i load_reg(const __m128i* addr)
{
    return _mm_load_si128(addr);
}

static __forceinline __m256i load_reg(const __m256i* addr)
{
    return _mm256_loadu_si256(addr);
}

static __forceinline void stream_reg(__m128i* adrr, __m128i& reg)
{
    _mm_stream_si128(adrr, reg);
}

static __forceinline void stream_reg(__m256i* adrr, __m256i& reg)
{
    _mm256_stream_si256(adrr, reg);
}

static __forceinline __m128i or_reg(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

static __forceinline __m256i or_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_or_si256(x, y);
}

static __forceinline __m128i xor_reg(const __m128i& x, const __m128i& y)
{
    return _mm_xor_si128(x, y);
}

static __forceinline __m256i xor_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_xor_si256(x, y);
}

static __forceinline __m128i and_reg(const __m128i& x, const __m128i& y)
{
    return _mm_and_si128(x, y);
}

static __forceinline __m256i and_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_and_si256(x, y);
}

static __forceinline __m128i srli_epi16(const __m128i& x, int count)
{
    return _mm_srli_epi16(x, count);
}

static __forceinline __m256i srli_epi16(const __m256i& x, int count)
{
    return _mm256_srli_epi16(x, count);
}

static __forceinline __m128i slli_epi16(const __m128i& x, int count)
{
    return _mm_slli_epi16(x, count);
}

static __forceinline __m256i slli_epi16(const __m256i& x, int count)
{
    return _mm256_slli_epi16(x, count);
}


static __forceinline void set1_epi8(__m128i& x, char v)
{
    x = _mm_set1_epi8(v);
}

static __forceinline void set1_epi8(__m256i& x, char v)
{
    x = _mm256_set1_epi8(v);
}

static __forceinline void set1_epi16(__m128i& x, int16_t v)
{
    x = _mm_set1_epi16(v);
}

static __forceinline void set1_epi16(__m256i& x, int16_t v)
{
    x = _mm256_set1_epi16(v);
}

static __forceinline __m128i add_epu16(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu16(x, y);
}

static __forceinline __m256i add_epu16(const __m256i& x, const __m256i& y)
{
    return _mm256_adds_epu16(x, y);
}

static __forceinline __m128i subs_epu8(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(x, y);
}

static __forceinline __m256i subs_epu8(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu8(x, y);
}

static __forceinline __m128i sub_epi16(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi16(x, y);
}

static __forceinline __m256i sub_epi16(const __m256i& x, const __m256i& y)
{
    return _mm256_sub_epi16(x, y);
}

static __forceinline __m128i mullo_epi16(const __m128i& x, const __m128i& y)
{
    return _mm_mullo_epi16(x, y);
}

static __forceinline __m256i mullo_epi16(const __m256i& x, const __m256i& y)
{
    return _mm256_mullo_epi16(x, y);
}

static __forceinline __m128i average(const __m128i& x, const __m128i& y)
{
    return _mm_avg_epu8(x, y);
}

static __forceinline __m256i average(const __m256i& x, const __m256i& y)
{
    return _mm256_avg_epu8(x, y);
}

template <typename T>
static __forceinline T
average(const T& w, const T& x, const T& y, const T& z)
{
    T one;
    set1_epi8(one, 0x01);
    T avg0 = average(w, x);
    T avg1 = average(y, z);
    T err0 = or_reg(xor_reg(w, x), xor_reg(y, z));
    T err1 = xor_reg(avg0, avg1);
    T mask = and_reg(and_reg(err0, err1), one);
    return subs_epu8(average(avg0, avg1), mask);
}

static __forceinline __m128i unpacklo_epi8(const __m128i& x, const __m128i& y)
{
    return _mm_unpacklo_epi8(x, y);
}

static __forceinline __m128i unpackhi_epi8(const __m128i& x, const __m128i& y)
{
    return _mm_unpackhi_epi8(x, y);
}

static __forceinline __m256i unpacklo_epi8(const __m256i& x, const __m256i& y)
{
    __m256i t0 = _mm256_unpacklo_epi8(x, y);
    __m256i t1 = _mm256_unpackhi_epi8(x, y);
    return _mm256_permute2x128_si256(t0, t1, 0x20);
}

static __forceinline __m256i unpackhi_epi8(const __m256i& x, const __m256i& y)
{
    __m256i t0 = _mm256_unpacklo_epi8(x, y);
    __m256i t1 = _mm256_unpackhi_epi8(x, y);
    return _mm256_permute2x128_si256(t0, t1, 0x31);
}

static __forceinline __m128i unpacklo_epi16(const __m128i& x, const __m128i& y)
{
    return _mm_unpacklo_epi16(x, y);
}

static __forceinline __m128i unpackhi_epi16(const __m128i& x, const __m128i& y)
{
    return _mm_unpackhi_epi16(x, y);
}

static __forceinline __m256i unpacklo_epi16(const __m256i& x, const __m256i& y)
{
    __m256i t0 = _mm256_unpacklo_epi16(x, y);
    __m256i t1 = _mm256_unpackhi_epi16(x, y);
    return _mm256_permute2x128_si256(t0, t1, 0x20);
}

static __forceinline __m256i unpackhi_epi16(const __m256i& x, const __m256i& y)
{
    __m256i t0 = _mm256_unpacklo_epi16(x, y);
    __m256i t1 = _mm256_unpackhi_epi16(x, y);
    return _mm256_permute2x128_si256(t0, t1, 0x31);
}

static __forceinline __m128i packus_epi16(const __m128i& x, const __m128i& y)
{
    return _mm_packus_epi16(x, y);
}

static __forceinline __m256i packus_epi16(const __m256i& x, const __m256i& y)
{
    //3,1,2,0 -> 0b11011000 = 216
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(x, y), 216);
}

template <typename T>
static __forceinline void cvtepu8_epi16x2(T& x, T& y)
{
    const T zero = xor_reg(x, x);
    y = unpackhi_epi8(x, zero);
    x = unpacklo_epi8(x, zero);
}

template <typename T>
static __forceinline void cvtepu8_epi16x2r(T& x, T& y)
{
    const T zero = xor_reg(x, x);
    y = unpackhi_epi8(zero, x);
    x = unpacklo_epi8(zero, x);
}


#endif
