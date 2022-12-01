from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='amusing',
    version='0.0.2',
    description='animating sheet music',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/leftgoes/aMusing',
    author='leftgoes (Ha Jong Kim)',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['audio2numpy', 'numpy', 'opencv-python', 'scipy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)