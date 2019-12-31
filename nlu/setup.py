from setuptools import setup

setup(
    name='ner_error_analysis',
    version='0.1',
    packages=['ext_utils'],
    package_dir={'': 'nlu'},
    url='https://github.com/ciaochiaociao/ner_error_analysis',
    license='',
    author='Chiao-Wei Hsu',
    author_email='tony790927@gmail.com',
    description='NER Error Analysis for column (conll format) dataset including CoNLL-2003, WNUT-2017, ...'
)
