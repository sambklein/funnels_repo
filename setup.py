from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="funnels",
    version='0.1',
    description="Experiments for studying distribution matching.",
    # long_description=long_description,
    long_description_content_type='text/markdown',
    author="Sam Klein",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    # Commented because it messes with the container
    # install_requires=[
    #     "matplotlib",
    #     "numpy",
    #     "tensorboard",
    #     "torch",
    #     "tqdm",
    #     "scipy",
    #     "pandas"
    # ],
    dependency_links=[],
)
