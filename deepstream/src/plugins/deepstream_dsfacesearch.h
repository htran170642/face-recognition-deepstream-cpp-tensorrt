/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _NVGSTDS_DSFACESEARCH_H_
#define _NVGSTDS_DSFACESEARCH_H_

#define CONFIG_GROUP_ENABLE "enable"
#define CONFIG_NVBUF_MEMORY_TYPE "nvbuf-memory-type"

#define NVDS_ELEM_DSFACESEARCH_ELEMENT "dsfacesearch"
#define CONFIG_GROUP_DSFACESEARCH "ds-facesearch"
// Refer to gst-dsfacesearch element source code for the meaning of these configs
#define CONFIG_GROUP_DSFACESEARCH_FULL_FRAME "full-frame"
#define CONFIG_GROUP_DSFACESEARCH_PROCESSING_WIDTH "processing-width"
#define CONFIG_GROUP_DSFACESEARCH_PROCESSING_HEIGHT "processing-height"
#define CONFIG_GROUP_DSFACESEARCH_BLUR_OBJECTS "blur-objects"
#define CONFIG_GROUP_DSFACESEARCH_UNIQUE_ID "unique-id"
#define CONFIG_GROUP_DSFACESEARCH_GPU_ID "gpu-id"

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif


typedef struct
{
  // Create a bin for the element only if enabled
  gboolean enable;
  // Struct members to store config / properties for the element
  gboolean full_frame;
  gint processing_width;
  gint processing_height;
  gboolean blur_objects;
  guint unique_id;
  guint gpu_id;
  // For nvvidconv
  guint nvbuf_memory_type;
} NvDsDsFaceSearchConfig;

// Struct to store references to the bin and elements
typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *pre_conv;
  GstElement *cap_filter;
  GstElement *elem_dsfacesearch;
} NvDsDsFaceSearchBin;

// Function to create the bin and set properties
gboolean
create_dsfacesearch_bin (NvDsDsFaceSearchConfig *config, NvDsDsFaceSearchBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSFACESEARCH_H_ */
