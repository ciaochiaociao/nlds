from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ner_error_analysis',
    version='0.2',
    packages=find_packages(),
    url='https://github.com/ciaochiaociao/ner_error_analysis',
    license='',
    author='Chiao-Wei Hsu',
    author_email='ciaochiaociao@gmail.com',
    description='NER Error Analysis for column (conll format) dataset including CoNLL-2003, WNUT-2017, ...',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    python_requires='>=3.6',
)
