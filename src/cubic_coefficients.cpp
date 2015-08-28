/*
cubic_coeffients.cpp

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
#include <cmath>

static inline int16_t round_to_short(const double d)
{
    return (int16_t)(d * 1024 + (d < 0.0 ? -0.5 : 0.5));
}

class MitchellNetravariCoefficients
{
    double p0, p2, p3, q0, q1, q2, q3;

    int16_t get_tap(const double distance)
    {
        double d = fabs(distance);
        if (d < 1.0) {
            return round_to_short(p0 + d * d * (p2 + d * p3));
        }
        return round_to_short(q0 + d * (q1 + d * (q2 + d * q3)));
    }

    void set_coeff_p(int16_t* coeff, int cplace)
    {
        if (cplace == 0) {
            coeff[0] = get_tap(-6.0 / 4);
            coeff[1] = get_tap(-2.0 / 4);
        } else if (cplace == 1) {
            coeff[0] = get_tap(-7.0 / 4);
            coeff[1] = get_tap(-3.0 / 4);
            coeff[2] = get_tap(1.0 / 4);
            coeff[3] = get_tap(5.0 / 4);
        } else if (cplace == 2) {
            coeff[0] = 0;
            coeff[1] = 2 * get_tap(-2.0 / 4);
            coeff[2] = 0;
            coeff[3] = 2 * get_tap(6.0 / 4);
        } else {
            coeff[0] = 2 * get_tap(-6.0 / 4);
            coeff[1] = 0;
            coeff[2] = 2 * get_tap(2.0 / 4);
            coeff[3] = 0;
        }
    }

    void set_coeff_i(int16_t* coeff, int cplace)
    {
        if (cplace == 0 || cplace == 3) {
            coeff[0] = get_tap(-12.0 / 8);
            coeff[1] = get_tap(-4.0 / 8);
        } else if (cplace == 1) {
            coeff[0] = get_tap(-14.0 / 8);
            coeff[1] = get_tap(-6.0 / 8);
            coeff[2] = get_tap(2.0 / 8);
            coeff[3] = get_tap(10.0 / 8);
            coeff[4] = get_tap(-14.0 / 8);
            coeff[5] = get_tap(-6.0 / 8);
            coeff[6] = get_tap(2.0 / 8);
            coeff[7] = get_tap(10.0 / 8);
        } else {
            coeff[0] = get_tap(-15.0 / 8);
            coeff[1] = get_tap(-7.0 / 8);
            coeff[2] = get_tap(1.0 / 8);
            coeff[3] = get_tap(9.0 / 8);
            coeff[4] = get_tap(-13.0 / 8);
            coeff[5] = get_tap(-5.0 / 8);
            coeff[6] = get_tap(3.0 / 8);
            coeff[7] = get_tap(11.0 / 8);
        }
    }

public:
    MitchellNetravariCoefficients(double b, double c)
    {   // taken from avisynth's resample code
        p0 = (  6. -  2. * b          ) / 6.0;
        p2 = (-18. + 12. * b +  6. * c) / 6.0;
        p3 = ( 12. -  9. * b -  6. * c) / 6.0;
        q0 = (        8. * b + 24. * c) / 6.0;
        q1 = (     - 12. * b - 48. * c) / 6.0;
        q2 = (        6. * b + 30. * c) / 6.0;
        q3 = (            -b -  6. * c) / 6.0;
    }

    void set_coeff(int16_t *coeff, bool interlaced, int cplace)
    {
        if (interlaced) {
            set_coeff_i(coeff, cplace);
        } else {
            set_coeff_p(coeff, cplace);
        }
    }
};

void set_cubic_coefficients(double b, double c, int16_t* array, bool interlaced, int cplace)
{
    auto mnc = MitchellNetravariCoefficients(b, c);
    mnc.set_coeff(array, interlaced, cplace);
}