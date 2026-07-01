import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    # 修复路径构建 - 直接使用相对路径
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=sources,  # 直接使用sources中的相对路径
    )
    return cuda_ext


setup(
    name="pcd utils",
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        make_cuda_ext(
            name="iou3d_nms_cuda",
            module="opencood.pcdet_utils.iou3d_nms",
            sources=[
                "iou3d_nms/src/iou3d_cpu.cpp",
                "iou3d_nms/src/iou3d_nms_api.cpp",
                "iou3d_nms/src/iou3d_nms.cpp",
                "iou3d_nms/src/iou3d_nms_kernel.cu",
            ],
        ),
        make_cuda_ext(
            name="roiaware_pool3d_cuda",
            module="opencood.pcdet_utils.roiaware_pool3d",
            sources=[
                "roiaware_pool3d/src/roiaware_pool3d.cpp",
                "roiaware_pool3d/src/roiaware_pool3d_kernel.cu",
            ],
        ),
        make_cuda_ext(
            name="pointnet2_stack_cuda",
            module="opencood.pcdet_utils.pointnet2.pointnet2_stack",
            sources=[
                "pointnet2/pointnet2_stack/src/pointnet2_api.cpp",
                "pointnet2/pointnet2_stack/src/ball_query.cpp",
                "pointnet2/pointnet2_stack/src/ball_query_gpu.cu",
                "pointnet2/pointnet2_stack/src/group_points.cpp",
                "pointnet2/pointnet2_stack/src/group_points_gpu.cu",
                "pointnet2/pointnet2_stack/src/sampling.cpp",
                "pointnet2/pointnet2_stack/src/sampling_gpu.cu",
                "pointnet2/pointnet2_stack/src/interpolate.cpp",
                "pointnet2/pointnet2_stack/src/interpolate_gpu.cu",
            ],
        ),
        make_cuda_ext(
            name="pointnet2_batch_cuda",
            module="opencood.pcdet_utils.pointnet2.pointnet2_batch",
            sources=[
                "pointnet2/pointnet2_batch/src/pointnet2_api.cpp",
                "pointnet2/pointnet2_batch/src/ball_query.cpp",
                "pointnet2/pointnet2_batch/src/ball_query_gpu.cu",
                "pointnet2/pointnet2_batch/src/group_points.cpp",
                "pointnet2/pointnet2_batch/src/group_points_gpu.cu",
                "pointnet2/pointnet2_batch/src/interpolate.cpp",
                "pointnet2/pointnet2_batch/src/interpolate_gpu.cu",
                "pointnet2/pointnet2_batch/src/sampling.cpp",
                "pointnet2/pointnet2_batch/src/sampling_gpu.cu",
            ],
        ),
    ],
)