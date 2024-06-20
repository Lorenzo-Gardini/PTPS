from setuptools import setup, find_packages

library_to_be_removed = ['pytest-cov', 'pytest']


def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Remove comments and empty lines
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements


setup(
    name='prepydf',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Lorenzo',
    author_email='l.gardo98@gmail.com',
    description='prepydf is a preprocessing python library for Pandas DataFrame',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lorenzo-Gardini/prepydf',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
