---
name: openscad
description: Design printable parts in OpenSCAD, render and check them before printing
status: tested
---
Used for the ES3C28P case and for parts of the flywheel launcher.
1. Write parametric `.scad` with named dimensions at the top. Measure the real part (calipers) instead of trusting a vendor drawing; the vendor numbers were off by 1.5 mm once.
2. Render: `openscad -o part.stl part.scad`. Preview images: `xvfb-run -a openscad -o view.png --imgsize=1200,900 --camera=... part.scad`. The version on the tested machines is 2021.01, so avoid newer-only features.
3. Overlap subtracted shapes slightly (epsilon of 0.01 to 0.1 mm) so no coincident faces remain.
4. Keep at least 2 mm of wall under engraved details. Check the cavity against the tallest component, not an average.
5. Check by exporting separate check models (clearance, pin positions) and always run a positive control: deliberately break a dimension and confirm the check fails. A check that cannot fail proves nothing.
6. Print with organic tree supports for overhangs. Slice a small test first; do not send a long print on an unproven setup.
