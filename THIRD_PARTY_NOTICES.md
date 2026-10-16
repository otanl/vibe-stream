# Third-party notices

This repository's code is Apache-2.0 (see `LICENSE`). It contains or depends on the following third-party work.
No model weights are included in the repository; `scripts/download_weights.py` fetches them from their publishers.

## Included code

### taesd (taesana architecture)

`src/vibe_stream/taesana.py` re-types the `F32Encoder` / `F32Decoder` architecture from
[madebyollin/taesd](https://github.com/madebyollin/taesd) `taesd.py`.

```
MIT License

Copyright (c) 2023 Ollin Boer Bohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies (not included)

| Component | Source | Licence as published |
|---|---|---|
| VIBE code (`vibe` package, sibling checkout) | [ai-forever/vibe](https://github.com/ai-forever/vibe) | Apache-2.0 |
| VIBE weights (CFG-distilled) | [iitolstykh/VIBE-Image-Edit-DistilledCFG](https://huggingface.co/iitolstykh/VIBE-Image-Edit-DistilledCFG) | the model card asks users to follow Sana's licence; no licence tag of its own |
| Sana 1.5 (the DiT VIBE starts from) | [NVlabs/Sana](https://github.com/NVlabs/Sana) | Apache-2.0 |
| Qwen3-VL-2B (VIBE's text encoder) | [Qwen](https://huggingface.co/Qwen) | Apache-2.0 |
| taesana weights | [madebyollin/taesd](https://github.com/madebyollin/taesd) | MIT |
| CLIP ViT-L/14 (evaluation only, `metrics.CLIPScore`) | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | MIT |
| DAVIS-2017 clips used for training and evaluation | [davischallenge.org](https://davischallenge.org/davis2017/code.html) | see the DAVIS terms; not redistributed here |

Check the publishers' current terms before redistributing weights or derived checkpoints (such as a trained student
LoRA or a TensorRT engine built from them).
