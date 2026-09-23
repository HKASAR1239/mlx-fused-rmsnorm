from setuptools import setup
from mlx import extension


setup(
    ext_modules=[extension.CMakeExtension("mlx_fused_rmsnorm._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    packages=["mlx_fused_rmsnorm"],
    zip_safe=False,
)
