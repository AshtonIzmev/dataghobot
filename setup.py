from setuptools import setup

# numpy and scipy are supposed to be already installed
# Anaconda2-2.4.1 contains everything you need

setup(name='dataghobot',
      py_modules=['dataghobot'],
      version='0.1.0',
      description='Dataghobot ersatz (Hyperparametrization, Feature transformation and generation)',
      url='https://github.com/AshtonIzmev/dataghobot',
      install_requires=['scikit-learn', 'pandas', 'hyperopt', 'keras', 'pymongo', 'xgboost', 'tqdm'],
      zip_safe=False)
