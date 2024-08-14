from setuptools import setup, find_namespace_packages

setup(name='angiographies',
      packages=find_namespace_packages(include=["angiographies", "angiographies.*"]),
      version='1.0.0',
      description='Nidus extraction.',
      url='https://github.com/CamilaGL/CVET-2023-Comparative',
      author='Camila Garcia',
      entry_points={
          'console_scripts': [
              'cvet_vmtknetwork = angiographies.skeletonisation.vmtknetwork:main',
              'cvet_thinning = angiographies.skeletonisation.orderedthinning:main',
              'cvet_nidusextractor = angiographies.nidus.skeextractor:main',
              'cvet_morph = angiographies.nidus.morphextractor:main',
          ],
     }
    )