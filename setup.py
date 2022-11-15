import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

try:
  with open(os.path.join(path, 'README.md')) as f:
    long_description = f.read()
except Exception as e:
  long_description = "Ganblr Toolbox"

setup(
    name = "ganblr",
    version = "0.1.0",
    keywords = ["ganblr", "tulip"],
    description = "Ganblr Toolbox",
    long_description = long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.5.0",
    license = "MIT Licence",

    url = "https://github.com/tulip-lab/ganblr-toolbox",
    author = "kae zhou",
    author_email = "kaezhou@gmail.com",

    packages = find_packages(),
    include_package_data = True,
    install_requires = ["numpy", "tensorflow>=2.3", "scikit-learn>=0.24", "pyitlab", "pgmpy"],
    platforms = "any",

    scripts = [],
    entry_points = {}
)