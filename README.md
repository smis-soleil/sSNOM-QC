# SNOM-QC a Streamlit web app for sSNOM/nanoFTIR QC

[![Application][application-shield]][application-link]
[![CC BY 4.0][license-shield]][license-link]

Description **_TODO_**

**Known limitations:** This app is not intended to replace data analysis

**_TODO_**



## Local installation

The application can be ran locally by executing the following commands, assuming git and Python (version 3.12) are installed:

```
git clone https://github.com/smis-soleil/snom-qc
cd anasys-python-tools-gui
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run source/app.py 
```

## Running tests

Run unit tests in the `snom-qc` conda environment:

```bash
mkdir -p /tmp/mpl /tmp/xdg-cache
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg-cache \
conda run --no-capture-output -n snom-qc python -m unittest discover -s tests -v
```

## Citing this work

This work is licensed under the GNU General Public License version 3.

**_TODO_**

[license-link]:       http://creativecommons.org/licenses/by/4.0/
[license-image]:      https://i.creativecommons.org/l/by/4.0/88x31.png
[license-shield]:     https://img.shields.io/badge/License-GNU_GPLv3-blue

[application-link]: https://snom-qc.streamlit.app
[application-shield]: https://img.shields.io/badge/Open_on_Streamlit-tomato
