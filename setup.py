import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='speclus4py',
    version='0.3.14',
    author='Marek Pecha',
    author_email='marek.pecha@vsb.cz',
    maintainer='Marek Pecha',
    maintainer_email='marek.pecha@vsb.cz',
    description='',
    long_description=long_description,
    long_description_content_type='',
    url='',
    packages=['speclus4py', 'speclus4py.tools'],
    package_data={'speclus4py': ['data']},
)
