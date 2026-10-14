# assets

Video clips are not tracked. The scripts expect:

- `davis_mp4/` — evaluation clips: DAVIS-2017 sequences re-encoded as mp4 at 24 fps (the 10 tracked in
  [dreamlite-stream](https://github.com/otanl/dreamlite-stream) `assets/davis_mp4/`; `parkour`, `libby`,
  `blackswan` and `dance-twirl` are the evaluation set, `blackswan` also the export check).
- `davis_heldout20_mp4/` — training clips: 25 further DAVIS-2017 sequences, none of which is in `davis_mp4/`.
  dreamlite-stream's `scripts/davis_to_mp4.py` builds both from the raw DAVIS-2017 480p frames
  ([download](https://davischallenge.org/davis2017/code.html)).

Any video works with `scripts/stream_video.py --input`; frames are centre-cropped to a square and resized.
