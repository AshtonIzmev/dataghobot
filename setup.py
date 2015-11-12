from setuptools import setup


setup(name='dataghobot',
      py_modules=['dataghobot'],
      version='0.1.0',
      description='Dataghobot ersatz (Hyperparametrization, Feature transformation and generation)',
      url='https://github.com/AshtonIzmev/dataghobot',
      install_requires=['pandas', 'sklearn', 'hyperopt', 'keras', 'wabbit_wappa', 'scipy', 'pymongo', 'xgboost'],
      zip_safe=False)
