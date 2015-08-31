#YV12To422
## YV12 to YV16/YUY2 converter for AviSynth2.6

YV12To422 is an avisynth filter plugin which based on YV12ToYUY2(ddcc.dll)
written by Kevin Stone(a.k.a tritical) and was written from scratch.

###Info:
    Convert YV12 clip to YV16/YUY2 using SSE2/AVX2.

    version 1.0.1


### Requirement:
    - avisynth2.60/avisynth+r1576 or later.
    - SSE2 capable CPU.
    - WindowsVista sp2 or later.
    - Visual C++ Redistributable Packages for Visual Studio 2013.

### Syntax:

    YV12To422(clip, bool "interlaced", int "itype", int "cplace", bool "lshift",
              bool "yuy2", bool "avx2", bool "threads", float "b", float "c")


    NOTE: these parameters may be changed later.
          (Sorry, I'm not enthusiastic about keeping backward compatibility.)


####    interlaced -

      Sets whether or not the input video is interlaced or progressive.

      default:  false


####    itype -

      Sets interpolation method. Possible settings:

          0 - duplicate (nearest neighbor)
          1 - linear interpolation
          2 - Mitchell-Netravali two-part cubic interpolation
                 (adjustable b/c parameters to adjust blurring/ringing)

      default:  2


####    cplace -

      Specifies vertical chroma placement.  Possible settings:

        progressive input (interlaced=false, progressive upsampling):

            0 - chroma is aligned with top line of each two line pair within the frame

                This would be the case if during 4:2:2 -> 4:2:0 conversion the chroma values of
                odd lines were simply dropped.

            1 - chroma is centered between lines of each two line pair within the frame
                (*** h261, h263, mpeg1, mpeg2, mpeg4, h264 standard progressive conversion)

                This would be the case if during 4:2:2 -> 4:2:0 conversion the chroma from every
                two line pair was averaged, or if an interlaced 4:2:2 -> 4:2:0 conversion was performed
                by using 75/25 averaging of top field pairs and 25/75 averaging of bottom field pairs.

            2 - chroma is aligned with top line of each two line pair within each field

                This would be the case if the 4:2:2 -> 4:2:0 conversion was performed by
                separating the fields, and then doing a 4:2:2 -> 4:2:0 conversion on each field
                by dropping odd line chroma values.

            3 - chroma is centered between lines of each two line pair within each field

                This would be the case if the 4:2:2 -> 4:2:0 conversion was performed by
                separating the fields, and then doing a 4:2:2 -> 4:2:0 conversion on each field
                by averaging chroma from every two line pair.

        interlaced input (interlaced=true, interlaced upsampling):

            0 - chroma is aligned with top line of each two line pair within each field

                This would be the case if the 4:2:2 -> 4:2:0 conversion was performed by
                separating the fields, and then doing a 4:2:2 -> 4:2:0 conversion on each field
                by dropping odd line chroma values.

            1 - chroma is centered between lines of each two line pair within each field

                This would be the case if the 4:2:2 -> 4:2:0 conversion was performed by
                separating the fields, and then doing a 4:2:2 -> 4:2:0 conversion on each field
                by averaging chroma from every two line pair.

            2 - top field chroma is 1/4 pixel below even lines in the top field, and
                bottom field chroma is 1/4 pixel above odd lines in the bottom field.
                (*** mpeg2, mpeg4, h264 standard interlaced conversion)

                This would be the case if the 4:2:2 -> 4:2:0 conversion was performed by
                averaging top field pairs using 75/25 weighting, and averaging bottom field
                pairs using 25/75 weighting.  This results in the same chroma placement as
                progressive cplace option 1.

            3 - U is aligned with top line of each two line pair within each field, and
                V is aligned with bottom line of each pair within each field.
                (*** DV-PAL standerd interlaced conversion)

        ** Progressive option 1 and interlaced option 2 are actually the same in terms of
           chroma placement.  If a progressive frame was converted to yv12 using interlaced
           method 2, then it is safe (actually better) to convert it to yuy2 using progressive
           method 1.  However, if a progressive frame was converted to yv12 using a different
           type of interlaced sampling, resulting in different chroma placement (such as those
           described by interlaced options 0 or 1), then it is best to convert it to yv16/yuy2
           using progressive upsampling, but with the cplace option that correctly specifies
           the positioning of the chroma.

      default:  1 (if interlaced = false)
                2 (if interlaced = true)


####    lshift -

      If set this to true, chroma placement will shift to 1/4 sample to the left.

        default: false


####    yuy2 -

      Sets whether or not the output video format is packed(YUY2) or planar(YV16).

      defaullt: true (YUY2 output)


####    avx2 -

      Sets whether AVX2 is used or not.

      default: false (use SSE2)

        ** Currentry, avisynth2.60 can't make memory alignment anything but 16bytes.
           Thus, if you use avisynth2.60, you shouldn't to set this true.
           avisynth+ has no problem.
           see https://github.com/AviSynth/AviSynthPlus/commit/ab4ea303b4ca78620c2ef90fdaad184bc18b7708


####    threads -

      When sets this to true, V-plain is processed with a different thread at the
      same time with U-plain.
      However, processing doesn't always become speedy by this.

      default: false (use single thread)


####    b / c -

      Adjusts properties of cubic interpolation (itype=2).  Same as Avisynth's BicubicResize filter.

      default:  0.0,0.75


### Lisence:

    GPLv2 or later

### Source code:

    https://github.com/chikuzen/YV12To422/

