# yapf: disable

from setuptools import setup, find_packages

packages = find_packages()
requirements = [
    "torch",
    "fairseq",
    # "transformers",
]

VERSION = {}  # type: ignore
with open("__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="sentiment_speech",
    version=VERSION["version"],
    description="Multimodal Speech Sentiment Recognition Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/JoungheeKim/",
    author="Joung Hee Kim",
    author_email="onlyl4youu@gmail.com",
    license="Apache-2.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    python_requires=">=3.6.0",
    setup_requires=["pytest-runner"],
    #tests_require=["pytest"],
    package_data={},
    include_package_data=True,
    dependency_links=[],
    zip_safe=False,
)