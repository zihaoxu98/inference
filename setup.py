import setuptools

setuptools.setup(
    name='inference',
    version='0.0.0',
    description='A toolkit for convenient likelihood definition and statistical inference',
    author='Zihao Xu',
    setup_requires=['pytest-runner'],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    url="https://github.com/xzh19980906/inference",
    zip_safe=False,
)
