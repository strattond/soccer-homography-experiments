# soccer-homography-experiments

This is a project for tinkering with and playing with homography
in support of analysis, coaching, video presentations and so on
for soccer.

Ultimately, the plan is to build out some degree of automatic 
tracking and people classification to find key events and drive
other things.

# Getting it setup

## Pre-requisites

* Python 3.12+
* [UV installed](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
* Windows or Linux (Mac may work, entirely untested)
* Footage you want to analyse!

## Preparing the environment

### Create the virtual environment

```pwsh
uv venv create .venv
.venv\Scripts\activate.ps1
```

### Install the dependencies

```pwsh
uv sync
```

## Running it

```pwsh
uv run python .\gui.py
```

