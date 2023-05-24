from setuptools import setup, find_packages

setup(
    name='cuda-tutorial',
    version='0.1',
    packages=find_packages(include=['.']),
    install_requires=[
          'torch', 'cupy', 'triton'
      ],
)