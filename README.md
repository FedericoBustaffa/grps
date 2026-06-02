# Agent-Based Evolutionary Rock-Paper-Scissors

An evolutionary approach to the **rock-paper-scissors** system in which three
species compete for space in a finite environment. The three species hunt each
other in a cyclic way, typically producing a particular equilibrium condition.

The extension to the original work consists in a evolutionary behavior inspired
by genetic algorthms, consisting in the addition of mating mechanism based on a
fitness value.

## Run

To run the code just create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

and install the package:

```bash
pip install .
```

Now is possible to run simulations through the `rps.py` file in `tests/` or run
the `notebook.ipynb` to see some plots.
