import setuptools

setuptools.setup(
    name='aptinf',
    version='0.0.0',
    description='Appletree inference',
    author='Appletree contributors, the XENON collaboration',
    setup_requires=['pytest-runner'],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    url="https://github.com/XENONnT/aptinf",
    zip_safe=False,
)
