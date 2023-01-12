#include "common.h"
#include <cuda_runtime_api.h>

gboolean get_converted_map(NvBufSurface *surface, gint idx, cv::Mat& frame_img){
  NvBufSurface surface_idx;
  surface_idx = *surface;
  surface_idx.numFilled = surface_idx.batchSize = 1;
  surface_idx.surfaceList = &(surface->surfaceList[idx]);

  NvBufSurfTransformParams nvbufsurface_params;
  NvBufSurface *dst_surface = NULL;
  NvBufSurfaceCreateParams nvbufsurface_create_params;
  cudaError_t cuda_err;
  cudaStream_t cuda_stream;
  gint create_result;
  NvBufSurfTransformConfigParams transform_config_params;
  NvBufSurfTransform_Error err;

  NvBufSurfTransformRect src_rect{0, 0, surface_idx.surfaceList->width, surface_idx.surfaceList->height};
  NvBufSurfTransformRect dst_rect{0, 0, surface_idx.surfaceList->width, surface_idx.surfaceList->height};

  nvbufsurface_params.src_rect = &src_rect;
  nvbufsurface_params.dst_rect = &dst_rect;
  nvbufsurface_params.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST;
  nvbufsurface_params.transform_filter = NvBufSurfTransformInter_Default;

  nvbufsurface_create_params.gpuId = surface_idx.gpuId;
  nvbufsurface_create_params.width = surface_idx.surfaceList->width;
  nvbufsurface_create_params.height = surface_idx.surfaceList->height;
  nvbufsurface_create_params.size = 0;
  nvbufsurface_create_params.isContiguous = true;
  nvbufsurface_create_params.colorFormat = NVBUF_COLOR_FORMAT_BGRA;
  nvbufsurface_create_params.layout = NVBUF_LAYOUT_PITCH;
#ifdef __aarch64__
  nvbufsurface_create_params.memType = NVBUF_MEM_DEFAULT;
#else
  nvbufsurface_create_params.memType = NVBUF_MEM_CUDA_PINNED;
#endif
  cuda_err = cudaSetDevice(surface_idx.gpuId);
  cuda_err = cudaStreamCreate(&cuda_stream);
  create_result = NvBufSurfaceCreate(&dst_surface, 1, &nvbufsurface_create_params);
  UNUSED(cuda_err);
  UNUSED(create_result);
  transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  transform_config_params.gpu_id = surface->gpuId;
  transform_config_params.cuda_stream = cuda_stream;
  err = NvBufSurfTransformSetSessionParams(&transform_config_params);

  NvBufSurfaceMemSet(dst_surface, 0, 0, 0);
  err = NvBufSurfTransform(&surface_idx, dst_surface, &nvbufsurface_params);

  if (err != NvBufSurfTransformError_Success){
    g_print("NvBufSurfTransform failed with error %d while converting buffer\n", err);
    return FALSE;
  }

  NvBufSurfaceMap(dst_surface, 0, 0, NVBUF_MAP_READ);
  NvBufSurfaceSyncForCpu(dst_surface, 0, 0);

  cv::Mat in_mat = cv::Mat(dst_surface->surfaceList->height, dst_surface->surfaceList->width, CV_8UC4,
                            dst_surface->surfaceList->mappedAddr.addr[0], dst_surface->surfaceList->pitch);

  cv::cvtColor(in_mat, frame_img, cv::COLOR_BGRA2BGR);

  NvBufSurfaceUnMap(dst_surface, 0, 0);
  NvBufSurfaceUnMap(&surface_idx, 0, 0);
  NvBufSurfaceDestroy(dst_surface);
  cudaStreamDestroy(cuda_stream);
  return TRUE;
}
