from distutils.core import setup

setup(
    version='1.0alpha',
    name='MLT',
    authro='Matthias Meidinger',
    packages=['MLT'],
    license='Apache License 2.0',
    url='https://github.com/Maddosaurus/MLT',
    install_requires=[
        'pandas',
        'matplotlib',
        'xgboost',
        'Keras',
        'pyod',
        'tensorflow',
        'natsort'
    ]
)