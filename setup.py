try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
config = {
'description': 'Crypto Trading project with CoinBase API',
'author': 'Sebastien Creoff',
'url': 'https://github.com/Sebastiencreoff/pythonTrading',
'download_url': 'https://github.com/Sebastiencreoff/pythonTrading',
'author_email': 'sebastien.creoff@gmail.com',
'version': '0.1',
'install_requires': ['nose'],
'packages': ['Trading'],
'scripts': [],
'name': 'pythonTrading'
}
setup(**config)