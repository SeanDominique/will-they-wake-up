from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='wtwu',
      version="0.0.2",
      description="Will They Wake Up model (model lifecycle)",
      license="",
      author="Briac, Elhadji, Mario, Sean",
      author_email="contact@lewagon.org",
      #url="https://github.com/SeanDominique/will-they-wake-up",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
