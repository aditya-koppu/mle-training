from setuptools import find_packages, setup

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

setup(
    name="tigertraining",
    version="0.0.1",
    description="Assignment",
    long_description=open("README.MD").read() + "\n\n" + open("README_IMP.MD").read(),
    url="",
    author="aditya_k",
    author_email="aditya.koppu@tigeranalytics.com",
    license="MIT",
    classifiers=classifiers,
    keywords="tiger",
    packages=find_packages(include=["script", "script.*"]),
    install_requires=[],
    setup_requires=["pytest-runner", "flake8"],
    tests_require=["pytest"],
)
