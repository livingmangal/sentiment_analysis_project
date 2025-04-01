from setuptools import setup, find_packages

setup(
    name="sentiment_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.3",
        "flask-cors>=4.0.0",
        "flask-limiter>=3.5.0",
        "torch>=2.6.0",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.3",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "gunicorn>=21.2.0",
    ],
) 