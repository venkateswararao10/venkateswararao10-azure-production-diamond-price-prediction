from setuptools import setup,find_packages
from typing import List

def getrequirements(filepath:str)->List[str]:
    f=open(filepath,'r')
    requirements = f.readlines()
    requirements=[i.replace('\n','') for i in requirements]
    if '-e .' in requirements:
      requirements.remove('-e .')
    f.close()
    return requirements
setup(
    name='priceprediction',
    version='0.0.1',
    author='venky',
    author_email='venky123@gmail.com',
    packages=find_packages(),
    install_requires=getrequirements('requirements.txt')
)