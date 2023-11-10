# EarthEarthquake Monitoring - Python Setup
- Python version: 3.11.5
- Install packages: `pip install -r requirements.txt` (first install seisbench from the submodule)

## seisbench
Since we adapt EQTransformer to only use one channel, the `seisbench` project is added as a submodule.  To install the *local* version of `seisbench` as opposed to the one on `pypi`, run the following from the project root with your virtual env active:
```bash
cd external/seisbench
pip install .
```
This installs the `seisbench` package inside `external/seisbench` into the current virtual environment.

After changes to the local `seisbench` package, `pip install .` needs to rerun for those changes to propagate to any place where `import seisbench` or similar is used.
