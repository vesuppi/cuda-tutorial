from setuptools import setup

setup(
    name='cuda-tutorial',
    version='0.1',
    packages=[],
    install_requires=[
          'torch', 'cupy', 'triton',
      ],
)
