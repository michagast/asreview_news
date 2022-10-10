from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='Dutch news feature extractor',
    version='0.1',
    description='Example classifier extension',
    url='https://github.com/asreview/asreview',
    author='ASReview team',
    author_email='asreview@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'sklearn',
        'asreview>=0.8',
        'transformers',
        'pyenchant',
        'nltk',
        'spacy',
        'num2words',
        'tensorflow',
        'spacy'
    ],
    entry_points={
        'asreview.models.classifiers': [
        ],
        'asreview.models.feature_extraction': [
            'news = asreviewcontrib.models.news_fe:News'
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/asreview/asreview/',
    },
)
