/*
  proc_to422.h

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


#ifndef YV12TO422_PROC_TO_422_H
#define YV12TO422_PROC_TO_422_H

#pragma warning(disable:4752)

#include <cstdint>

using proc_to422 = void (__stdcall *)(
    const int aligned_width, const int height, const uint8_t* srcp,
    uint8_t* dstp, int src_pitch, int dst_pitch);

proc_to422 get_proc_chroma(int itype, int cplace, bool interlaced, bool avx2);

proc_to422 get_proc_horizontal_shift(bool use_avx2);

static inline int aligned_size(int x, int align)
{
    return ((x + align - 1) / align) * align;
}


#endif
