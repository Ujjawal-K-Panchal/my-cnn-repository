# CNN-Showcase
A Repository for showcasing CNN development, training, tuning and deployment.
Please look at the slide show for relevant details.


## Contents:
1. [Docs/](./docs/): All documentation present here. Can be \LaTeX files.
2. [src/](./src/): source code folder.
3.  


## Setup:
1. Upgrade pip: `python -n pip install --upgrade pip`
2. Install wheel: `pip install wheel`
3. Make vsenv: `python -m venv venv4cnn`
4. Activate virtual environment:
	- Windows: `venv4cnn\Scripts\activate`
	- Mac/Linux: `source venv4vnn/bin/activate`
5. Install requirements: `pip install -r requirements.txt`

## Usage:
(If virtualenv is not active, we recommend following Setup's step 4.)
1. cd into source programs folder: `cd src`
2. run unit tests:
	- model unit test: `python model.py`
	- pipeline funcs unit test: `python pipeline.py`
3. run main file to train network: `python main.py` 

## Deployment Instructions:
Deployment Instructions.