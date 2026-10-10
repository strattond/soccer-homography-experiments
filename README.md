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

In **Data Maintenance** > **Persons**, choose **Add generic people** to create the reusable Main Referee, Assistant Referees, Opposition Keepers, and Opposition Players if they do not already exist. With a registered clip loaded, **Add Generic** assigns those people to the clip's match as participants: referees get the Referee role, keepers get Away Goalkeeper, and players get Away Player. Since participations are match-level, these generic participants are available to every clip from that match.

When a clip is loaded, existing detection and tracking Parquet chunks in `tracking/` are restored into the canvases and Tracks table. Detection and tracking boxes are stored in source-frame coordinates; tracked rows also include the bounding-box centroid. **Crops** uses those coordinates directly against the decoded source frame for full-resolution crops. The box overlays are redrawn when the canvas is panned or zoomed.

Tracker IDs are short-term motion tracks, not stable player identities. Track history is kept unchanged in Parquet, while `TrackSegment` records optional participant assignments for inclusive frame ranges. Consecutive frame runs initialize segments; saved database segments override those initial ranges. The Tracks table shows the participant assigned to the selected frame's segment. **Next** and **Previous** navigate segment boundaries, and **Split** divides a segment after the current frame while carrying its participant assignment into both halves. The minimap shows unassigned segments in red and assigned segments in blue. Live Preview marker colors follow the assigned participant's role at each frame, or Unknown when no participant is assigned.

The database API can create a placeholder participation for an unidentified opposition player and associate it with part of a tracker history:

```python
from soccer_homography.db import (
    TrackSegmentDB,
    createPlaceholderParticipation,
    listTrackSegments,
    replaceTrackSegments,
    upsertTrackSegment,
)

placeholder = createPlaceholderParticipation(conn, match_id)
upsertTrackSegment(conn, TrackSegmentDB(clip_id, track_id, placeholder.person_id, 0, 149))
segments = listTrackSegments(conn, clip_id)
```

Segments may also be saved with `person_id=None` while identity is unknown. `replaceTrackSegments` atomically replaces all segments for one clip/track. Roles are edited on the associated `PersonParticipation`. Recreate an existing DuckDB database when adopting this schema; automatic migrations are not supported. Set **Crops per track segment** in **Model Options** (default 6, configurable from 1 to 50), then click **Crops** to collect that many samples for every unassigned segment. Crop images are cached per clip as `{track_id}_{segment_start}_{frame}.png`, allowing samples from separate segments of one tracker ID to remain isolated. Crop collection prioritizes shared video frames across segments and spreads each segment's selected frames across its lifespan. Homographies can be saved to the database explicitly with **Save Homography**; the initial save uses a locked-off camera range.

The **VLM** button analyzes the selected track's cached crops with the model selected in **Image Options**. **VLM** (the default) uses Moondream and its editable multi-line identification prompt, saved to `identificationPrompt.txt` in the repository root. **Clip** uses zero-shot CLIP classification over the six participation roles and reports its per-crop confidence. Both modes show each crop's role and a tally for every role; the most-voted role can be saved to the assigned participant's `PersonParticipation`, while a tie is reported without assigning a role.
