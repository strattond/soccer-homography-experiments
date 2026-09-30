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

## Importing Squadi match details

In the app, open **Data Maintenance**, select the **Matches** tab, and choose **Import**. Select a `matchDetails.json` file and then the division to import. If a `results.json` file is alongside it, the importer uses its match results to set home and away team names. Dates are stored as timestamps, parsed from Squadi's `YYYYMMDDHHmm` format. Re-importing updates existing matches and participations and adds any newly listed players.

To load a registered database clip, choose **Clip** in the main window and select a clip from the list. The **Clip Participants** tab shows the people associated with that clip's match.

When a clip is loaded, existing detection and tracking Parquet chunks in `tracking/` are restored into the canvases and Tracks table. Detection and tracking boxes are stored in source-frame coordinates; **Crops** uses those coordinates directly against the decoded source frame for full-resolution crops. The box overlays are redrawn when the canvas is panned or zoomed.

After tracks are available, assign a participant or role from the Tracks table; assignments are saved to DuckDB for the selected clip. Click **Crops** to collect up to six cached samples for each track without an assigned participant. Crop collection samples shared video frames first to serve multiple tracks per decoded frame, then falls back to per-track sampling only for tracks missed by those shared frames. Homographies can be saved to the database explicitly with **Save Homography**; the initial save uses a locked-off camera range.

The **VLM** button analyzes the selected track's cached crops with the model selected in **Image Options**. **VLM** (the default) uses Moondream and its editable multi-line identification prompt, saved to `identificationPrompt.txt` in the repository root. **Clip** uses zero-shot CLIP classification over the six participation roles and reports its per-crop confidence. Both modes show each crop's role and a tally for every role; the most-voted role can be saved to the track, while a tie is reported without assigning a role.
