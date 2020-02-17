from setuptools import setup, find_packages

setup(
    name='enriched_ph',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    description='Enriched persistent barcode via cohomology.',
    author='Zixuan Cang',
    author_email='cangzx@gmail.com'
)
