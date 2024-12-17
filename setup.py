from setuptools import setup, find_packages

setup(
    name='pyksvd',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    author='Mathias Grau and Antoine Martinez',
    author_email='mathias4grau@gmail.com',
    description='A Python implementation of the K-SVD algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mathias-grau/PyKSVD',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
