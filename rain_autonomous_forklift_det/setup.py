import os
from glob import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

package_name = 'rain_autonomous_forklift_det'

def make_cuda_ext(name, module, sources, **kwargs):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        **kwargs
    )
    return cuda_ext

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rain',
    maintainer_email='rain@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rain_autonomous_forklift_det = rain_autonomous_forklift_det.pointcloud_object_detector:main',
        ],
    },
    cmdclass={
        'build_ext': BuildExtension,
    },
    ext_modules=[
        make_cuda_ext(
            name='points_in_boxes_gpu_cuda',
            module='rain_autonomous_forklift_det.utils.points_in_boxes_gpu',
            sources=[
                'src/points_in_boxes_gpu.cpp',
                'src/points_in_boxes_gpu_kernel.cu',
            ]
        ),
    ],
)
