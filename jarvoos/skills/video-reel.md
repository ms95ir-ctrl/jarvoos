---
name: video-reel
description: Make a short narrated video from designed frames, voiceover and ffmpeg (no AI video model)
status: tested
---
Pipeline used for the Edubahn reel (9 shots, about 51 s, 1080x1920).
1. Write a shot-by-shot plan first: narrator line, on-screen text, visual, and how each shot hands over to the next. Keep claims honest; no invented testimonials or numbers.
2. Produce frames per shot as images (`frames/s1_%04d.jpg`, 30 fps) from HTML/SVG or code-drawn graphics.
3. Per shot: `ffmpeg -framerate 30 -i frames/s1_%04d.jpg -c:v libx264 -pix_fmt yuv420p s1.mp4`.
4. Join shots with `xfade` transitions (fade, slideleft, slideup) in one `-filter_complex`; account for the overlap when computing the total length.
5. Voiceover: one TTS file per shot, timed to the shot length, mixed with simple scripted sound effects (`amix`), then muxed over the video. A poster frame: `-vframes 1`.
6. Check the result by pulling frames out of the final file and looking at them, not by trusting the script.
What this is not: it does not generate imagery with an AI model.
