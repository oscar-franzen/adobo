# adobo's setup script.
# OF; Sept 2019

from setuptools import setup, find_packages, Extension

setup(
    name='adobo',
    version=open('adobo/VERSION').read().replace('\n',''),
    description='An analysis framework for single cell gene expression data.',
    author='Oscar Franzén',
    author_email='p.oscar.franzen@gmail.com',
    include_package_data=True,
    packages=['adobo', 'adobo.glm', 'adobo.irlbpy'],
    url='https://github.com/oscar-franzen/adobo',
    license='LICENSE',
    long_description=open('README.md').read(),
    install_requires=[
        'pandas >= 0.25.0',
        'numpy >= 1.17.0',
        'scikit-learn >= 0.21.3',
        'leidenalg >= 0.7.0',
        'python-igraph >= 0.7.1',
        'scipy >= 1.3.0',
        'umap-learn >= 0.3.9',
        'statsmodels >= 0.10.1',
        'matplotlib >= 3.1.1',
        'seaborn >= 0.9.0',
        'psutil >= 5.4.2'
        'datatable >= 0.9.0'
    ],
    ext_modules=[Extension('pdf', sources = ['adobo/libs/pdf.c'],
                           extra_compile_args=['-fPIC','-lm'])]
)
