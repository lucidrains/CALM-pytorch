from setuptools import setup, find_packages

setup(
  name = 'CALM-Pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'CALM - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/CALM-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'composing LLMs'
  ],
  install_requires = [
    'beartype',
    'einops>=0.7.0',
    'pytorch-custom-utils>=0.0.9',
    'torch>=2.0',
    'x-transformers>=1.27.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
