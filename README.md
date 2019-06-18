## Project guidelines

### Project Title

#ground2float
Identifying grounding zones from ICESat-2

### Collaborators on this project

 - Kiya Riverman (project lead)
 - Johan Nilsson (data person)
 - Bertie Miles
 - Sridhar Anandakrishnan
 - Matt Siegfried
 - Susheel Adusumilli
 - Maya Becker

### The problem

Grounding zones are very important but hard to identify in space. One tool with which to do this is altimetry, which can be used to locate slope breaks or detect tidal signals. We will develop tools to identify grounding zones from ICESat-2.


### Application Example

One location where these tools could be applied is Rutford Ice Stream. In this location, ICESat-2 tracks are roughly along flow. There are also two tracks over Thwaites that might contain valuable data--tracks 1079 (December 8, 2018) and 576 (November 11, 2018).

### Sample data

ICESat, REMA, earlier DEMs, ice-flow direction

### Specific Questions

 - Find the grounding zone in three or four places (e.g., Siple Coast ice streams, Foundation Ice Stream, etc.)
 - Test ability to identify moving grounding lines (Bungenstockrücken Ice Plain?)
 - Test grounding-zone detection ability by comparing with known InSAR-derived grounding line positions
 - Determine where the slope-break method is a viable method to find the grounding zone

### Existing methods

Slope-break technique, InSAR, and altimetry (tidal flexure/elevation changes).

### Proposed methods/tools

We hope to develop a tool to calculate along-flow slopes from not-along-flow lines. We would also like to develop a tool to compare the ICESat-2 results with REMA or another data set. Perhaps we could also use CATS2008 model outputs to estimate the tidal elevation for the ICESat-2 track in question. This could be especially helpful in finding the grounding zone where the slope-break estimate is not as useful.

Our plan is to test out the tool on a single track in the Thwaites region and then eventually apply it to all of the tracks with data in a Thwaites bounding box.

### Background reading

Fricker et al. (2009), Bindschadler et al. (2011)

## Data files

aws  s3 cp s3://pangeo-data-upload-oregon/icesat2/ground2float/ .
