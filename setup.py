import os
from setuptools import find_packages, setup


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('tiago_rl/assets')

setup(name='tiago_rl',
      version='0.1',
      description='TIAGo Reinforcement Learning Environments',
      author='Luca Lach',
      author_email='llach@techfak.uni-bielefeld.de',
      url='https://github.com/llach/tiago_rl',
      packages=[package for package in find_packages() if package.startswith("tiago_rl")],
      package_data={'tiago_rl': extra_files},
)