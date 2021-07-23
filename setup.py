from setuptools import find_packages, setup

setup(name='tiago_rl',
      version='0.1',
      description='TIAGo Reinforcement Learning Environments',
      author='Luca Lach',
      author_email='luca.lach@pal-robotics.com',
      url='https://github.com/llach/tiago_rl',
      packages=[package for package in find_packages() if package.startswith("tiago_rl")],
     )