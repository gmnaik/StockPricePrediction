from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(filepath:str) -> List[str]:
    '''
    This function will return a list of requirements. i.e libraries that needs to be installed
    '''
    requirements=[]
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()                              # Read the requirements.txt file line by line
        print("requirements before",requirements)
        requirements = [req.replace("\n","") for req in requirements]    # Remove the "\n" that gets added in the list due to readlines() so that only package name will be present
        print("requirements after",requirements)
        
        if HYPEN_E_DOT in requirements:                                  # In order to remove -e . that will be added in requirements list from requirements.txt file. 
            requirements.remove(HYPEN_E_DOT)
        else:
            pass
    
    print("requirements final",requirements)
    return requirements
        
setup(
    name='StockPricePrediction',
    version='0.0.1',
    author='Goutam',
    author_email='gmnaik96@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
