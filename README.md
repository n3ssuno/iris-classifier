# IRIS Virtual Patent Marking Pages Classifier
Tool to help a human being to classify a list of potential VPM pages into several possible categories. Part of the IRIS project.

The classifier is written in Python, using the PyQt5 library.

It creates a GUI browser that shows sequentially one of the detected pages.

You can interact with the browser with the mouse and you can also use the numerical pad of the keyboard to select one of the categories.

Once you have chosen the right category for a page the software moves to the next page.

## Setup the classifier
The best is to 
1. Install [Git](https://git-scm.com/)
2. Clone this repository with ``git clone https://github.com/n3ssuno/iris-classifier.git``
3. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
4. Create an environment with
    * ``conda create -n iris-vpm-pages-classifier python=3.9``
	* ``conda activate iris-vpm-pages-classifier``
	* ``pip install -r requirements.txt``
5. If you need to use the pre-classifier, you must also install a headless browser with the following command<br>
	``playwright install chromium``<br>
	Note: the code has been tested with Chromium v857950 but the last version of the browser will be installed

If needed, you can add ``iris_utils`` as a submodule with (this should be already in place after point 2 above):
* ``git submodule add https://github.com/n3ssuno/iris-utils.git iris_utils``
* ``git commit -m "Add iris-utils submodule"``
* ``git push``

### GUI classifier on WSL2
1. Install ``qt5-default`` on the WSL2 distro
2. Install X410 on Windows (the free alternatives did not work for me) and select ``Allow Public Access`` from its menu
3. Add the following lines into the ``~/.bashrc`` file of the WSL2 distro (before the bunch of code about Conda)<br>
``export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0.0``<br>
Instead, do not add ``export LIBGL_ALWAYS_INDIRECT=1`` as adviced in many online guides.

## Pre-processing
Before you start to classify the pages by hand, you must run ``pre-classify.py`` to automatically classify some pages.
This script will create a file with five main categories: cases that are (a) very likely true positives; (b) very likely false positives; (c) maybe positive; (d) maybe negative; (e) unknown.

The first two cases are automatically classified. For the second two, a hint is provided and the person is required to choose if the page is actually a VPM page or not. The last case is left to the person, without any hint.

To use it you need a bunch of software that is as easy to install on GNU/Linux as hard to have on MS-Windows. The advice is, therefore, to use a GNU/Linux machine (the instructions that follow are for Debian GNU/Linux) or use WSL2 (to run the GUI classifier from WSL2 is not trivial but possible; follow the instructions here below).
1. Install [Tesseract](https://tesseract-ocr.github.io/) with<br>
``sudo apt install tesseract-ocr``
2. Install [Poppler](https://poppler.freedesktop.org/)<br>
``sudo apt install poppler-utils``

To run the automatic classifier, please run<br>
``python pre-classify.py -I data/scraping_results.jsonl data/websites_to_exclude.txt -o data/pre_classified.jsonl``

## Populate the database
Once the data have been analyzed by the pre-classifier, you must use its output to populate a database that will be used by the classifier. To do so, please run<br>
``python write-database.py -I data/scraping_results.jsonl data/pre_classified.jsonl -o data/database.json``

If you want to split the data in sub-databased, so that more than one person can have her/his own data to classify, you can run<br>
``python write-database.py -I data/scraping_results.jsonl data/pre_classified.jsonl -o data/database.json -O N``<br>
where ``N`` is the number of files that you want to generate.

Note: you cannot overwrite the database once created (you can only update it, if not using the specific commands of [Flata](https://github.com/harryho/flata)). If you want to do so, you must delete the written files and re-run the script.

## Run the classifier
1. Remember, each time, to activate the conda environment created in the setup phase with ``conda activate iris-vpm-pages-classifier``
2. Run ``python classify.py -i data/database.json``

## Acknowledgements
The authors thank the EuroTech Universities Alliance for sponsoring this work. Carlo Bottai was supported by the European Union's Marie Skłodowska-Curie programme for the project Insights on the "Real Impact" of Science (H2020 MSCA-COFUND-2016 Action, Grant Agreement No 754462).
