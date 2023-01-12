#ifndef __CONVERTED_MAP_H__
#define __CONVERTED_MAP_H__
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glib-object.h>

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

gboolean get_converted_map(NvBufSurface *surface, gint idx, cv::Mat& frame_img);
#endif