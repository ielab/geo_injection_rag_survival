from setuptools import setup, find_packages

setup(
    name='llm-rankers',
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/ielab/llm-rankers',
    license='Apache 2.0',
    author='Shengyao Zhuang',
    author_email='s.zhuang@uq.edu.au',
    description='Pointwise, Listwise, Pairwise and Setwise Document Ranking with Large Language Models.',
    python_requires='>=3.8',
    install_requires=[
        "transformers>=4.31.0",
    ]
)