import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='JamInTune-alexberrian',
     version='0.0.0',
     description='Tune an audio recording to match standard Western piano key frequencies',
     long_description=long_description,
     long_description_content_type="text/markdown",
     author='Alex Berrian',
     author_email='ajberrian@ucdavis.edu',
     url='https://www.github.com/alexberrian/JamInTune/',
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD 3-Clause \"New\" or \"Revised\" License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.7',
     install_requires=['numpy>=1.16.4',
                       'librosa>=0.8.0',
                       'soundfile>=0.10.3.post1',
                       'weightedstats>=0.4.1',
                       'sox>=1.4.1'],
)