Machine Learning on Proteomics Data to guide library expansion
----------------------------------------------------------------

let's go through the Jupyter Notebook located in vignettes/main.py

### To go through the notebook report:
---------------------------------------

* install your virtual env with python: `python3 -m venv ml` then `source ml/bin/activate`
* OR with conda (recommended) `conda create -n ML python=3.12.3` then `conda activate ML`
* install dependencies: `pip install -r requirements.txt`
* install Vina: `conda install conda-forge::vina`
* install OpenBabel: `conda install -c conda-forge openbabel`
* `python -m ipykernel install --user --name ML --display-name "Python (ML)"`