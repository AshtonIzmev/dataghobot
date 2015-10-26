from setuptools import setup


setup(name='dataghobot',
      version='0.1.0',
      description='Dataghobot ersatz (Hyperparametrization, Feature transformation and generation)',
      url='https://github.com/AshtonIzmev/dataghobot',
      install_requires=['pandas', 'sklearn', 'hyperopt', 'keras', 'wabbit_wappa'],
      zip_safe=False)
