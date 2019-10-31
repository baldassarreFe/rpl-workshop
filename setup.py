from setuptools import setup, find_packages

tests_require = [
    'pytest',
]

setup(
    name='workshop',
    version='0.0.1',
    description='Workshop on GPU and slurm usage at RPL, KTH',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
    },
)
