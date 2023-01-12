/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * There are two threads in the optimized code. input thread and Processing thread.
 * The pre-procesing as required by the algorithm like scaling and color
 * conversion of data is done in input thread. This is done using NvBufSurfTransform's
 * batch conversion APIs to improve performance. The processing of data using custom
 * algorithm and parsing the output and  metadata attachment is done in separate processing
 * thread.
 *
 * There are two queues used for buffering and transferring data between thread:
 * Process_queue and buf_queue. Process_queue is used to send filled batched data to
 * process thread and buf_queue is used to get return empty processed buffers from
 * process thread to input thread.  Two buffers are used in a ping pong manner between
 * the two threads for parallel processing.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>

#include "gstdsfacesearch_optimized.h"

#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include <thread>

GST_DEBUG_CATEGORY_STATIC (gst_dsfacesearch_debug);
#define GST_CAT_DEFAULT gst_dsfacesearch_debug
#define USE_EGLIMAGE 1

#ifdef WITH_OPENCV
//enable to write transformed cvmat to files
// #define GST_DSFACESEARCH_DEBUG

#endif

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_GPU_DEVICE_ID
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })


/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 112
#define DEFAULT_PROCESSING_HEIGHT 112
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsfacesearch_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsfacesearch_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsfacesearch_parent_class parent_class
G_DEFINE_TYPE (GstDsFaceSearch, gst_dsfacesearch, GST_TYPE_BASE_TRANSFORM);

static void gst_dsfacesearch_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsfacesearch_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dsfacesearch_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsfacesearch_start (GstBaseTransform * btrans);
static gboolean gst_dsfacesearch_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_dsfacesearch_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_dsfacesearch_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);

static void attach_metadata_object (GstDsFaceSearch * dsfacesearch,
    NvDsObjectMeta * obj_meta, DsFaceSearchOutput * output);

static gpointer gst_dsfacesearch_output_loop (gpointer data);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsfacesearch_class_init (GstDsFaceSearchClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsfacesearch_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsfacesearch_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsfacesearch_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsfacesearch_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsfacesearch_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_dsfacesearch_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_dsfacesearch_generate_output);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width",
          "Processing Width",
          "Width of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_WIDTH, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height",
          "Processing Height",
          "Height of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame",
          "Full frame",
          "Enable to process full frame or disable to process objects detected"
          "by primary detector", DEFAULT_PROCESS_FULL_FRAME, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing",
          1, NVDSFACESEARCH_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsfacesearch_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsfacesearch_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsFaceSearch plugin",
      "DsFaceSearch Plugin",
      "Process a 3rdparty facesearch algorithm on objects / full frame",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsfacesearch_init (GstDsFaceSearch * dsfacesearch)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsfacesearch);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsfacesearch->unique_id = DEFAULT_UNIQUE_ID;
  dsfacesearch->processing_width = DEFAULT_PROCESSING_WIDTH;
  dsfacesearch->processing_height = DEFAULT_PROCESSING_HEIGHT;
  dsfacesearch->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  dsfacesearch->gpu_id = DEFAULT_GPU_ID;
  dsfacesearch->max_batch_size = DEFAULT_BATCH_SIZE;
  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsfacesearch_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsfacesearch->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      dsfacesearch->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      dsfacesearch->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      dsfacesearch->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dsfacesearch->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      dsfacesearch->max_batch_size = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsfacesearch_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsfacesearch->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, dsfacesearch->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, dsfacesearch->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, dsfacesearch->process_full_frame);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsfacesearch->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, dsfacesearch->max_batch_size);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the process thread
 */
static gboolean
gst_dsfacesearch_start (GstBaseTransform * btrans)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (btrans);
  std::string nvtx_str;

  g_print("Initializing HTTP client...");
  std::string host = "localhost";
  std::string port = "18080";
  std::string url = "/recognize";
  dsfacesearch->http_client =
    std::unique_ptr<HttpClient>(
      new HttpClient(host, port, url));

#ifdef WITH_OPENCV
  // OpenCV mat containing RGB data
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif
  NvBufSurfaceCreateParams create_params;

  nvtx_str = "GstNvDsFaceSearch: UID=" + std::to_string(dsfacesearch->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (dsfacesearch->gpu_id),
      "Unable to set cuda device");

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsfacesearch->cuda_stream),
      "Could not create cuda stream");

#ifdef WITH_OPENCV
  if (dsfacesearch->inter_buf)
    NvBufSurfaceDestroy (dsfacesearch->inter_buf);
  dsfacesearch->inter_buf = NULL;
#endif

  /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
   * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
  create_params.gpuId = dsfacesearch->gpu_id;
  create_params.width = dsfacesearch->processing_width;
  create_params.height = dsfacesearch->processing_height;
  create_params.size = 0;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;
#ifdef __aarch64__
  create_params.memType = NVBUF_MEM_DEFAULT;
#else
  create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
#endif

#ifdef WITH_OPENCV
  if (NvBufSurfaceCreate (&dsfacesearch->inter_buf, dsfacesearch->max_batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer for dsfacesearch");
    goto error;
  }
#endif

  /* Create process queue and cvmat queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  dsfacesearch->process_queue = g_queue_new ();
  dsfacesearch->buf_queue = g_queue_new ();

#ifdef WITH_OPENCV
  /* Push cvmat buffer twice on the buf_queue which will handle the
   * different processing speed between input thread and process thread
   * cvmat queue is used for getting processed data from the process thread*/
  for (int i = 0; i < 2; i++) {
    // CV Mat containing interleaved RGB data.
    cvmat = new cv::Mat[dsfacesearch->max_batch_size];

    for (guint j = 0; j < dsfacesearch->max_batch_size; j++) {
      cvmat[j] =
          cv::Mat (dsfacesearch->processing_height, dsfacesearch->processing_width,
          CV_8UC3);
    }

    if (!cvmat)
      goto error;

    g_queue_push_tail (dsfacesearch->buf_queue, cvmat);
  }

  GST_DEBUG_OBJECT (dsfacesearch, "created CV Mat\n");
#else
  for (int i = 0; i < 2; i++) {
    if (NvBufSurfaceCreate (&inter_buf, dsfacesearch->max_batch_size,
          &create_params) != 0) {
      GST_ERROR ("Error: Could not allocate internal buffer for dsfacesearch");
      goto error;
    }

    g_queue_push_tail (dsfacesearch->buf_queue, inter_buf);
  }
#endif

  /* Set the NvBufSurfTransform config parameters. */
  dsfacesearch->transform_config_params.compute_mode =
      NvBufSurfTransformCompute_Default;
  dsfacesearch->transform_config_params.gpu_id = dsfacesearch->gpu_id;

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  dsfacesearch->batch_insurf.surfaceList =
      new NvBufSurfaceParams[dsfacesearch->max_batch_size];
  dsfacesearch->batch_insurf.batchSize = dsfacesearch->max_batch_size;
  dsfacesearch->batch_insurf.gpuId = dsfacesearch->gpu_id;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  dsfacesearch->transform_params.src_rect =
      new NvBufSurfTransformRect[dsfacesearch->max_batch_size];
  dsfacesearch->transform_params.dst_rect =
      new NvBufSurfTransformRect[dsfacesearch->max_batch_size];
  dsfacesearch->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  dsfacesearch->transform_params.transform_flip = NvBufSurfTransform_None;
  dsfacesearch->transform_params.transform_filter =
      NvBufSurfTransformInter_Default;

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  dsfacesearch->process_thread =
      g_thread_new ("dsfacesearch-process-thread", gst_dsfacesearch_output_loop,
      dsfacesearch);

  dsfacesearch->nvtx_domain = nvtx_domain_ptr.release ();

  return TRUE;
error:

  delete[]dsfacesearch->transform_params.src_rect;
  delete[]dsfacesearch->transform_params.dst_rect;
  delete[]dsfacesearch->batch_insurf.surfaceList;

  if (dsfacesearch->cuda_stream) {
    cudaStreamDestroy (dsfacesearch->cuda_stream);
    dsfacesearch->cuda_stream = NULL;
  }
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_dsfacesearch_stop (GstBaseTransform * btrans)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (btrans);

#ifdef WITH_OPENCV
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif

  g_mutex_lock (&dsfacesearch->process_lock);

  /* Wait till all the items in the queue are handled. */
  while (!g_queue_is_empty (dsfacesearch->process_queue)) {
    g_cond_wait (&dsfacesearch->process_cond, &dsfacesearch->process_lock);
  }

#ifdef WITH_OPENCV
  while (!g_queue_is_empty (dsfacesearch->buf_queue)) {
    cvmat = (cv::Mat *) g_queue_pop_head (dsfacesearch->buf_queue);
    delete[]cvmat;
    cvmat = NULL;
  }
#else
  while (!g_queue_is_empty (dsfacesearch->buf_queue)) {
    inter_buf = (NvBufSurface *) g_queue_pop_head (dsfacesearch->buf_queue);
    if (inter_buf)
      NvBufSurfaceDestroy (inter_buf);
    inter_buf = NULL;
  }
#endif
  dsfacesearch->stop = TRUE;

  g_cond_broadcast (&dsfacesearch->process_cond);
  g_mutex_unlock (&dsfacesearch->process_lock);

  g_thread_join (dsfacesearch->process_thread);

#ifdef WITH_OPENCV
  if (dsfacesearch->inter_buf)
    NvBufSurfaceDestroy (dsfacesearch->inter_buf);
  dsfacesearch->inter_buf = NULL;
#endif

  if (dsfacesearch->cuda_stream)
    cudaStreamDestroy (dsfacesearch->cuda_stream);
  dsfacesearch->cuda_stream = NULL;

  delete[]dsfacesearch->transform_params.src_rect;
  delete[]dsfacesearch->transform_params.dst_rect;
  delete[]dsfacesearch->batch_insurf.surfaceList;

#ifdef WITH_OPENCV
  GST_DEBUG_OBJECT (dsfacesearch, "deleted CV Mat \n");
#endif

  GST_DEBUG_OBJECT (dsfacesearch, "ctx lib released \n");

  g_queue_free (dsfacesearch->process_queue);

  g_queue_free (dsfacesearch->buf_queue);

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsfacesearch_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dsfacesearch->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (dsfacesearch->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio and fills data for batched conversation */
static GstFlowReturn
scale_and_fill_data(GstDsFaceSearch * dsfacesearch,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    gdouble & ratio, gint input_width, gint input_height)
{

  gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
  gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
  gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
  gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);
  guint dest_width = MIN(src_width, RECOGNITION_INPUT_WIDTH);
  guint dest_height = MIN(src_height, RECOGNITION_INPUT_HEIGHT);

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
#ifdef __aarch64__
  if (ratio <= 1.0 / 16 || ratio >= 16.0) {
    // Currently cannot scale by ratio > 16 or < 1/16 for Jetson
    return GST_FLOW_ERROR;
  }
#endif

  /* We will first convert only the Region of Interest (the entire frame or the
   * object bounding box) to RGB and then scale the converted RGB frame to
   * processing resolution. */
  GST_DEBUG_OBJECT (dsfacesearch, "Scaling and converting input buffer\n");

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  dsfacesearch->batch_insurf.surfaceList[dsfacesearch->batch_insurf.numFilled] = *src_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  dsfacesearch->transform_params.src_rect[dsfacesearch->batch_insurf.numFilled] = {
  (guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  dsfacesearch->transform_params.dst_rect[dsfacesearch->batch_insurf.numFilled] = {
  0, 0, dest_width, dest_height};

  dsfacesearch->batch_insurf.numFilled++;

  return GST_FLOW_OK;
}

static gboolean
convert_batch_and_push_to_process_thread (GstDsFaceSearch * dsfacesearch,
    GstDsFaceSearchBatch * batch)
{

  NvBufSurfTransform_Error err;
  NvBufSurfTransformConfigParams transform_config_params;
  std::string nvtx_str;
#ifdef WITH_OPENCV
  cv::Mat in_mat;
#endif

  // Configure transform session parameters for the transformation
  transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  transform_config_params.gpu_id = dsfacesearch->gpu_id;
  transform_config_params.cuda_stream = dsfacesearch->cuda_stream;

  err = NvBufSurfTransformSetSessionParams (&transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err),
        (NULL));
    return FALSE;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(dsfacesearch->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(dsfacesearch->nvtx_domain, &eventAttrib);

  g_mutex_lock (&dsfacesearch->process_lock);

  /* Wait if buf queue is empty. */
  while (g_queue_is_empty (dsfacesearch->buf_queue)) {
    g_cond_wait (&dsfacesearch->buf_cond, &dsfacesearch->process_lock);
  }

#ifdef WITH_OPENCV
  /* Pop a buffer from the element's buf queue. */
  batch->cvmat = (cv::Mat *) g_queue_pop_head (dsfacesearch->buf_queue);
#else
  /* Pop a buffer from the element's buf queue. */
  batch->inter_buf = (NvBufSurface *) g_queue_pop_head (dsfacesearch->buf_queue);
  dsfacesearch->inter_buf = batch->inter_buf;
#endif

  g_mutex_unlock (&dsfacesearch->process_lock);

  //Memset the memory
  for (uint i = 0; i < dsfacesearch->batch_insurf.numFilled; i++)
    NvBufSurfaceMemSet (dsfacesearch->inter_buf, i, 0, 0);

  /* Batched tranformation. */
  err = NvBufSurfTransform (&dsfacesearch->batch_insurf, dsfacesearch->inter_buf,
      &dsfacesearch->transform_params);

  nvtxDomainRangePop (dsfacesearch->nvtx_domain);

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer",
            err), (NULL));
    return FALSE;
  }

  // Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
  // algorithm can handle padded RGBA data.
  for (guint i = 0; i < dsfacesearch->batch_insurf.numFilled; i++) {
    // Map the buffer so that it can be accessed by CPU
    if (NvBufSurfaceMap (dsfacesearch->inter_buf, i, 0, NVBUF_MAP_READ) != 0) {
      GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
          ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }
    // sync mapped data for CPU access
    NvBufSurfaceSyncForCpu (dsfacesearch->inter_buf, i,0);

#ifdef WITH_OPENCV
    in_mat =
        cv::Mat (dsfacesearch->processing_height, dsfacesearch->processing_width,
        CV_8UC4,dsfacesearch->inter_buf->surfaceList[i].mappedAddr.addr[0],
      dsfacesearch->inter_buf->surfaceList[i].pitch);

#if (CV_MAJOR_VERSION >= 4)
  cv::cvtColor (in_mat, batch->cvmat[i], cv::COLOR_RGBA2BGR);
#else
  cv::cvtColor (in_mat, batch->cvmat[i], CV_RGBA2BGR);
#endif

#ifdef GST_DSFACESEARCH_DEBUG
  static guint cnt = 0;
  cv::imwrite("out_" + std::to_string (cnt) + ".jpeg", batch->cvmat[i]);
  cnt++;
#endif
#endif

    if (NvBufSurfaceUnMap (dsfacesearch->inter_buf, i,0)) {
      GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("%s:buffer unmap to be accessed by CPU failed", __func__), (NULL));
      return FALSE;
    }

#ifdef __aarch64__
  // To use the converted buffer in CUDA, create an EGLImage and then use
  // CUDA-EGL interop APIs
  if (USE_EGLIMAGE) {
    if (NvBufSurfaceMapEglImage (dsfacesearch->inter_buf, 0) != 0) {
      GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("%s:buffer map eglimage failed", __func__), (NULL));
      return FALSE;
    }
    // dsfacesearch->inter_buf->surfaceList[0].mappedAddr.eglImage
    // Use interop APIs cuGraphicsEGLRegisterImage and
    // cuGraphicsResourceGetMappedEglFrame to access the buffer in CUDA

    // Destroy the EGLImage
    NvBufSurfaceUnMapEglImage (dsfacesearch->inter_buf, 0);
  }
#endif
  }

  /* Push the batch info structure in the processing queue and notify the process
   * thread that a new batch has been queued. */
  g_mutex_lock (&dsfacesearch->process_lock);

  g_queue_push_tail (dsfacesearch->process_queue, batch);
  g_cond_broadcast (&dsfacesearch->process_cond);

  g_mutex_unlock (&dsfacesearch->process_lock);

  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsfacesearch_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstDsFaceSearchBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;
  std::unique_ptr < GstDsFaceSearchBatch > batch = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;
  gdouble scale_ratio = 1.0;
  guint num_filled = 0;

  dsfacesearch->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(dsfacesearch->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(dsfacesearch->nvtx_domain, &eventAttrib);

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsfacesearch));

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  num_filled = batch_meta->num_frames_in_batch;

  // Using object crops as input to the algorithm. The objects are detected by
  // the primary detector
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_obj = NULL;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);

    // printf("plugin frame idx: %d\n", frame_meta->frame_num);

#if 0
  cv::Mat bgr_frame;
  if (!get_converted_map(in_surf, frame_meta->batch_id, bgr_frame))
    continue;
  cv::imwrite(std::to_string(frame_meta->frame_num)+".jpg", bgr_frame);
  guint obj_idx = 0;
  for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
      l_obj = l_obj->next) {
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
    float left = obj_meta->detector_bbox_info.org_bbox_coords.left;
    float top = obj_meta->detector_bbox_info.org_bbox_coords.top;
    float width = obj_meta->detector_bbox_info.org_bbox_coords.width;
    float height = obj_meta->detector_bbox_info.org_bbox_coords.height;
    printf("%f %f %f %f \n", left, top, width, height);  
    try {
      cv::Rect2f bbox = cv::Rect2f(left, top, width, height);

      std::string file_name = std::to_string(frame_meta->frame_num) +"_"+ std::to_string((guint)obj_idx)+".jpg";
      cv::imwrite(file_name, bgr_frame.clone()(bbox));
    }
    catch (std::exception &e) {
      printf("%s\n", e.what());
    }
    
    obj_idx++;
  }
  cv::imwrite(std::to_string(frame_meta->frame_num)+".jpg", bgr_frame);
#endif

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT
        * since it will cause hardware scaling issues. */
      if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
          obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
        continue;

      // Crop and scale the object maintainig aspect ratio
      if (scale_and_fill_data (dsfacesearch,
              in_surf->surfaceList + frame_meta->batch_id,
              &obj_meta->rect_params, scale_ratio,
              dsfacesearch->video_info.width,
              dsfacesearch->video_info.height) != GST_FLOW_OK) {
        // Error in conversion, skip processing on object. */
        continue;
      }

      if (batch == nullptr) {
        batch.reset (new GstDsFaceSearchBatch);
        batch->push_buffer = FALSE;
        batch->inbuf = inbuf;
        batch->inbuf_batch_num = dsfacesearch->current_batch_num;
        batch->nvtx_complete_buf_range = buf_process_range;
      }

      /* Adding a frame to the current batch. Set the frames members. */
      GstDsFaceSearchFrame frame;
      frame.scale_ratio_x = scale_ratio;
      frame.scale_ratio_y = scale_ratio;
      frame.obj_meta = obj_meta;
      frame.frame_meta =
          nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
      frame.frame_num = frame.frame_meta->frame_num;
      frame.batch_index = i;
      frame.input_surf_params = in_surf->surfaceList + i;
      batch->frames.push_back (frame);

      i++;

      // Convert batch and push to process thread
      if (batch->frames.size () == dsfacesearch->max_batch_size
          || i == num_filled) {
        if (!convert_batch_and_push_to_process_thread (dsfacesearch,
                batch.get ())) {
          return GST_FLOW_ERROR;
        }
        /* Batch submitted. Set batch to nullptr so that a new GstDsFaceSearchBatch
          * structure can be allocated if required. */
        i = 0;
        batch.release ();
        dsfacesearch->batch_insurf.numFilled = 0;
      }
    }
  }

    /* Submit a non-full batch. */
  if (batch) {
    if (!convert_batch_and_push_to_process_thread (dsfacesearch, batch.get ())) {
      return GST_FLOW_ERROR;
    }
    batch.release ();
    dsfacesearch->batch_insurf.numFilled = 0;
  }

  nvtxDomainRangeEnd(dsfacesearch->nvtx_domain, buf_process_range);

  /* Queue a push buffer batch. This batch is not inferred. This batch is to
   * signal the process thread that there are no more batches
   * belonging to this input buffer and this GstBuffer can be pushed to
   * downstream element once all the previous processing is done. */
  buf_push_batch = new GstDsFaceSearchBatch;
  buf_push_batch->inbuf = inbuf;
  buf_push_batch->push_buffer = TRUE;
  buf_push_batch->nvtx_complete_buf_range = buf_process_range;

  g_mutex_lock (&dsfacesearch->process_lock);
  /* Check if this is a push buffer or event marker batch. If yes, no need to
   * queue the input for inferencing. */
  if (buf_push_batch->push_buffer) {
    /* Push the batch info structure in the processing queue and notify the
     * process thread that a new batch has been queued. */
    g_queue_push_tail (dsfacesearch->process_queue, buf_push_batch);
    g_cond_broadcast (&dsfacesearch->process_cond);
  }
  g_mutex_unlock (&dsfacesearch->process_lock);

  flow_ret = GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_dsfacesearch_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (btrans);
  return dsfacesearch->last_flow_ret;
}

// In case of an actual processing library, processing on data wil be completed
// in this function and output will be returned
DsFaceSearchOutput *
DsFaceSearchProcess (float score, std::string& label)
{
    DsFaceSearchOutput *out =
        (DsFaceSearchOutput*)calloc (1, sizeof (DsFaceSearchOutput));

    out->numObjects = 1;
    out->object[0] = (DsFaceSearchObject)
    {
        score, ""
    };
    // Set the object label
    snprintf (out->object[0].label, 64, label.c_str ());

    return out;
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstDsFaceSearch * dsfacesearch, NvDsObjectMeta * obj_meta,
    DsFaceSearchOutput * output)
{
  if (output->numObjects == 0)
    return;
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

  NvDsClassifierMeta *classifier_meta =
      nvds_acquire_classifier_meta_from_pool (batch_meta);

  classifier_meta->unique_component_id = dsfacesearch->unique_id;

  NvDsLabelInfo *label_info =
      nvds_acquire_label_info_meta_from_pool (batch_meta);
  std::string str = std::string(output->object[0].label) + " " + std::to_string(output->object[0].score);
  g_strlcpy (label_info->result_label, str.c_str(), MAX_LABEL_SIZE);

  nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
  nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);

  nvds_acquire_meta_lock (batch_meta);
  NvOSD_TextParams & text_params = obj_meta->text_params;
  NvOSD_RectParams & rect_params = obj_meta->rect_params;

  /* Below code to display the result */
  // Set black background for the text
  // display_text required heap allocated memory
  if (text_params.display_text) {
    gchar *conc_string = g_strconcat (text_params.display_text, " ",
        output->object[0].label, NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
  } else {
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    // std::string str = std::string(output->object[0].label) + " - " + std::to_string(output->object[0].score);
    text_params.display_text = g_strdup (output->object[0].label);
    // Font face, size and color
    text_params.font_params.font_name = (char *) "Serif";
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_dsfacesearch_output_loop (gpointer data)
{
  GstDsFaceSearch *dsfacesearch = GST_DSFACESEARCH (data);
  DsFaceSearchOutput *output;
  NvDsObjectMeta *obj_meta = NULL;
  gdouble scale_ratio = 1.0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;

  nvtx_str =
      "gst-dsfacesearch_output-loop_uid=" + std::to_string (dsfacesearch->unique_id);

  g_mutex_lock (&dsfacesearch->process_lock);

  /* Run till signalled to stop. */
  while (!dsfacesearch->stop) {
    std::unique_ptr < GstDsFaceSearchBatch > batch = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (dsfacesearch->process_queue)) {
      g_cond_wait (&dsfacesearch->process_cond, &dsfacesearch->process_lock);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstDsFaceSearchBatch *)
        g_queue_pop_head (dsfacesearch->process_queue));
    g_cond_broadcast (&dsfacesearch->process_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    g_mutex_unlock (&dsfacesearch->process_lock);

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(dsfacesearch->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp (batch->inbuf,
          GST_ELEMENT_NAME (dsfacesearch));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (dsfacesearch),
          batch->inbuf);
      if (dsfacesearch->last_flow_ret != flow_ret) {
        switch (flow_ret) {
            /* Signal the application for pad push errors by posting a error message
             * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (dsfacesearch, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name (flow_ret), flow_ret));
            break;
          default:
            break;
        }
      }
      dsfacesearch->last_flow_ret = flow_ret;
      g_mutex_lock (&dsfacesearch->process_lock);
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(dsfacesearch->nvtx_domain, &eventAttrib);

    /* For each frame attach metadata output. */
    for (guint i = 0; i < batch->frames.size (); i++) {
      GstDsFaceSearchFrame & frame = batch->frames[i];

        obj_meta = frame.obj_meta;

        /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT
         * since it will cause hardware scaling issues. */
        if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
            obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
          continue;

        // Process the object crop to obtain label
#ifdef WITH_OPENCV
        // output = DsFaceSearchProcess ();
        
        // Encode
        std::vector<uchar> buf;
        cv::imencode(".jpg", batch->cvmat[i], buf, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 100});
        std::string s = std::string(buf.begin(), buf.end());
        std::string res = dsfacesearch->http_client->send(s);
#else
        // output = DsFaceSearchProcess (dsfacesearch->dsfacesearchlib_ctx,
        //     (unsigned char *)batch->inter_buf->surfaceList[i].mappedAddr.addr[0]);
#endif
        
        json j;

        // Buffer to json
        j = json::parse(res);
        if (j.size() > 0) {
            // std::cout << "Prediction: " << j["userId"] << " " << j["similarity"] << "\n";
            float score = j["similarity"].get<float>();;
            std::string label = j["userId"].get<std::string>();
            LOG_MESSAGE(2, LOG_LEVEL_INFO,
                        "Sourceid: %d, prediction:  %s, similarity: %f", frame.frame_meta->source_id, label.c_str(), score);
            output = DsFaceSearchProcess (score, label);
        } else {
            std::cout << "No prediction\n";
        }

        // Attach labels for the object
        attach_metadata_object (dsfacesearch, obj_meta, output);

        free (output);
    }

    g_mutex_lock (&dsfacesearch->process_lock);

#ifdef WITH_OPENCV
    g_queue_push_tail (dsfacesearch->buf_queue, batch->cvmat);
#else
    g_queue_push_tail (dsfacesearch->buf_queue, batch->inter_buf);
#endif
    g_cond_broadcast (&dsfacesearch->buf_cond);

    nvtxDomainRangePop (dsfacesearch->nvtx_domain);
  }
  g_mutex_unlock (&dsfacesearch->process_lock);

  return nullptr;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsfacesearch_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsfacesearch_debug, "dsfacesearch", 0,
      "dsfacesearch plugin");

  return gst_element_register (plugin, "dsfacesearch", GST_RANK_PRIMARY,
      GST_TYPE_DSFACESEARCH);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsfacesearch,
    DESCRIPTION, dsfacesearch_plugin_init, "6.1", LICENSE, BINARY_PACKAGE,
    URL)
