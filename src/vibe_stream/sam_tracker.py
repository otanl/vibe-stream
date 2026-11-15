"""SAM 2.1 hiera-tiny (Apache-2.0) as a streaming tracker for local edits: transformers' Sam2VideoModel weights and
modules, run frame by frame with one CUDA graph per frame.

`load_sam2(res)` loads the model for an input resolution below its native 1024 (a multiple of 32): the config's image
size, feature sizes and memory-attention RoPE grid are set for `res` and Hiera interpolates its position embedding by
itself; no retraining. The masks at 512 match those at 1024 closely for whole objects (IoU 0.9 and up) and less for
small parts (0.61-0.97).

transformers' `Sam2VideoModel.forward` spends most of a frame on the host on Windows (about 45 ms per object on an
RTX 3090 Ti, nearly the same at a quarter of the pixels): dictionary bookkeeping, a pageable host-to-device copy per
frame that waits for the GPU, and some hundred kernel launches. `LeanTracker` keeps the tracking state in fixed-size
buffers on the GPU (one object per tracker, as transformers runs them) and runs the same computation as
transformers' streaming path for a tracked frame:
  memory    the conditioning frame's memory (temporal slot 0) and the 6 most recent frames' (slots 6 ... 1), with
            their spatial and temporal position encodings
  pointers  the conditioning frame's object pointer and those of the 15 most recent frames, temporal positions
            (frame - their frame) / 15 through the sine encoding and its projection
  step      vision encoder -> memory attention -> SAM head (no prompt, multimask, best IoU) -> memory encoder
Frame 0 (the box prompt) goes through transformers' own session; its outputs seed the buffers. Each count of
memories and pointers (16 while the buffers fill, then the steady one) has its own CUDA graph; `precapture` makes them
all on a blank frame beforehand (about 4 s, 0.15-0.5 GB), so a clip runs graphs from frame 1. The masks match
transformers' (IoU 0.97-0.999); a frame takes 9 ms at 512 and 13 ms at 640 on an RTX 3090 Ti (10 / 18 ms on an RTX
4060 Ti).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SAM2 = "facebook/sam2.1-hiera-tiny"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_sam2(res: int = 512, dtype=torch.bfloat16, model: str = SAM2):
    """(model, processor) for input resolution `res` (a multiple of 32), on the GPU."""
    from transformers import Sam2VideoConfig, Sam2VideoModel, Sam2VideoProcessor
    if res % 32:
        raise ValueError("res must be a multiple of 32")
    cfg = Sam2VideoConfig.from_pretrained(model)
    if res != 1024:
        f = res // 16
        cfg.image_size = res
        cfg.prompt_encoder_config.image_size = res
        cfg.vision_config.backbone_config.image_size = [res, res]
        cfg.vision_config.backbone_feature_sizes = [[4 * f, 4 * f], [2 * f, 2 * f], [f, f]]
        cfg.memory_attention_rope_feat_sizes = [f, f]
    sam = Sam2VideoModel.from_pretrained(model, config=cfg, dtype=dtype).to("cuda").eval()
    proc = Sam2VideoProcessor.from_pretrained(model)
    proc.target_size = res
    return sam, proc


def pixels(frames_u8: torch.Tensor, res: int, dtype=torch.bfloat16) -> torch.Tensor:
    """(N, H, W, 3) uint8 on the GPU -> SAM's normalised input (N, 3, res, res)."""
    x = frames_u8.permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).to(dtype)


class LeanTracker:
    """One object followed through a clip (module docstring). `start(frame, [box])` on frame 0, then call it on each
    further frame; both return the mask logits at 256 x 256 (1, 1, 256, 256)."""

    def __init__(self, sam, proc, res: int, dtype=torch.bfloat16, graph: bool = True):
        from transformers.models.sam2_video.modeling_sam2_video import get_1d_sine_pe
        self.sam, self.proc, self.res, self.dt, self.use_graph = sam, proc, res, dtype, graph
        self.sine_pe = get_1d_sine_pe
        self.n_mem = sam.num_maskmem                              # 7: conditioning frame + 6 recent
        self.n_ptr = sam.config.max_object_pointers_in_encoder    # 16: conditioning frame + 15 recent
        self.multimask = sam._use_multimask(False, None)
        # one CUDA graph per (memories, pointers) count; they never run at the same time, so they share one pool
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle() if graph else None
        self.seeded = False

    @torch.inference_mode()
    def start(self, frame_u8: torch.Tensor, boxes) -> torch.Tensor:
        """frame_u8 (S, S, 3) uint8 on the GPU, boxes [[x0, y0, x1, y1]] in its pixels (one box)."""
        if len(boxes) != 1:
            raise ValueError("one object per LeanTracker (transformers' SAM head indexes one object per call)")
        size = frame_u8.shape[0]
        sess = self.proc.init_video_session(inference_device="cuda", dtype=self.dt)
        self.proc.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=[1], input_boxes=[boxes],
                                                  original_size=[size, size])
        o = self.sam(inference_session=sess, frame=pixels(frame_u8[None], self.res, self.dt)[0])
        out = sess.output_dict_per_obj[0]["cond_frame_outputs"][0]
        self.cond_mem = out["maskmem_features"]                  # (HW, 1, C)
        mem_pos = out["maskmem_pos_enc"]                         # the same sine encoding for every frame
        self.cond_ptr = out["object_pointer"]                    # (1, 256)
        self.B = 1
        tpe = self.sam.memory_temporal_positional_encoding      # (7, 1, 1, 64); slot 0 (conditioning) uses [-1]
        self.mem_pos_slot = [mem_pos + tpe[(s - 1) % self.n_mem] for s in range(self.n_mem)]
        self.recent_mem, self.recent_ptr = [], []                # newest last
        self.t = 0
        self.seeded = False      # graphs are kept across clips; the steady one's buffers are refilled once full
        return o.pred_masks

    def _step(self, px, mem, mem_pos, ptr, ptr_offsets):
        """px (1, 3, R, R); mem / mem_pos (N_mem * HW, B, 64); ptr (N_ptr, B, 256); ptr_offsets (N_ptr,) float32
        (frame distance / 15). Returns (mask logits (B, 1, 256, 256), new memory (HW, B, 64), new pointer (B, 256),
        object score logits (B, 1))."""
        sam, B = self.sam, self.B
        feats, pos, _, _ = sam.get_image_features(px)
        hw = sam.backbone_feature_sizes
        high_res = [x.permute(1, 2, 0).reshape(B, x.size(2), *s) for x, s in zip(feats[:-1], hw[:-1])]
        # object pointers: temporal sine encoding, split into 4 tokens of the memory width
        pe = sam.temporal_positional_encoding_projection_layer(self.sine_pe(ptr_offsets, dim=sam.hidden_dim).to(ptr.dtype))
        pe = pe.unsqueeze(1).expand(-1, B, sam.mem_dim)
        split = sam.hidden_dim // sam.mem_dim
        ptr_tok = ptr.reshape(-1, B, split, sam.mem_dim).permute(0, 2, 1, 3).flatten(0, 1)
        pe = pe.repeat_interleave(split, dim=0)
        memory = torch.cat([mem, ptr_tok.to(mem.dtype)], 0)
        memory_pos = torch.cat([mem_pos, pe.to(mem_pos.dtype)], 0)
        cond = sam.memory_attention(current_vision_features=feats[-1], current_vision_position_embeddings=pos[-1],
                                    memory=memory, memory_posision_embeddings=memory_pos,
                                    num_object_pointer_tokens=ptr_tok.shape[0])
        h, w = hw[-1]
        pix_feat = cond.squeeze(1).permute(0, 2, 1).reshape(B, sam.hidden_dim, h, w)
        out = sam._single_frame_forward(pixel_values=None, image_embeddings=high_res + [pix_feat],
                                        multimask_output=self.multimask)
        new_mem, _ = sam._encode_new_memory(current_vision_feats=feats[-1], pred_masks_high_res=out.high_res_masks,
                                            object_score_logits=out.object_score_logits, is_mask_from_pts=False)
        return out.pred_masks, new_mem, out.object_pointer, out.object_score_logits

    def _inputs(self, offsets: bool = True):
        """Memory / pointer tensors in transformers' order: conditioning frame, then memories of frames t-6 ... t-1
        (oldest first), pointers of frames t-1 ... t-15 (newest first); the pointers' temporal offsets only if asked
        (a host-to-device copy, which waits for the GPU)."""
        k = len(self.recent_mem)
        slots = [0] + list(range(k, 0, -1))                     # temporal slots of cond, t-k, ..., t-1
        mem = torch.cat([self.cond_mem] + [self.recent_mem[-s] for s in range(k, 0, -1)], 0)
        mem_pos = torch.cat([self.mem_pos_slot[s] for s in slots], 0)
        ptr = torch.stack([self.cond_ptr] + [self.recent_ptr[-d] for d in range(1, len(self.recent_ptr) + 1)], 0)
        if not offsets:
            return mem, mem_pos, ptr, None
        offs = [self.t] + list(range(1, len(self.recent_ptr) + 1))
        ptr_offsets = torch.tensor(offs, dtype=torch.float32).div_(self.n_ptr - 1).to("cuda", non_blocking=True)
        return mem, mem_pos, ptr, ptr_offsets

    @torch.inference_mode()
    def __call__(self, frame_u8: torch.Tensor) -> torch.Tensor:
        """Track one more frame (S, S, 3) uint8 on the GPU."""
        self.t += 1
        px = pixels(frame_u8[None], self.res, self.dt)
        key = (len(self.recent_mem), len(self.recent_ptr))
        if not self.use_graph:
            mem, mem_pos, ptr, offs = self._inputs()
            masks, new_mem, new_ptr, _ = self._step(px, mem, mem_pos, ptr, offs)
            self._append(new_mem, new_ptr)
            return masks
        g = self.graphs.get(key) or self._capture(key, px)
        if key != (self.n_mem - 1, self.n_ptr - 1):
            # buffers filling: this frame's inputs into the graph's buffers, outputs cloned (the pool is shared)
            mem, mem_pos, ptr, _ = self._inputs(offsets=False)
            g["mem"].copy_(mem)
            g["mem_pos"].copy_(mem_pos)
            g["ptr"].copy_(ptr)
            g["px"].copy_(px)
            g["offs"][0] = self.t / (self.n_ptr - 1)
            g["graph"].replay()
            masks, new_mem, new_ptr = (x.clone() for x in g["out"][:3])
            self._append(new_mem, new_ptr)
            return masks
        if not self.seeded:
            mem, mem_pos, ptr, _ = self._inputs(offsets=False)
            g["mem"].copy_(mem)
            g["mem_pos"].copy_(mem_pos)
            g["ptr"].copy_(ptr)
        self.seeded = True
        g["px"].copy_(px)
        g["offs"][0] = self.t / (self.n_ptr - 1)                # the conditioning frame's distance grows
        g["graph"].replay()
        # roll: memories oldest first (drop t-6, append the new one), pointers newest first (insert at 1)
        hw = self.cond_mem.shape[0]
        m, p = g["mem"], g["ptr"]
        m[hw:-hw].copy_(m[2 * hw:].clone())
        m[-hw:].copy_(g["out"][1])
        p[2:].copy_(p[1:-1].clone())
        p[1].copy_(g["out"][2])
        return g["out"][0]

    def _append(self, new_mem, new_ptr) -> None:
        self.recent_mem = (self.recent_mem + [new_mem])[-(self.n_mem - 1):]
        self.recent_ptr = (self.recent_ptr + [new_ptr])[-(self.n_ptr - 1):]

    def _capture(self, key, px):
        mem, mem_pos, ptr, _ = self._inputs(offsets=False)
        offs = torch.arange(key[1] + 1, dtype=torch.float32, device="cuda")
        offs[0] = self.t
        g = dict(px=px.clone(), mem=mem.clone(), mem_pos=mem_pos.clone(), ptr=ptr.clone(), offs=offs / (self.n_ptr - 1))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step(g["px"], g["mem"], g["mem_pos"], g["ptr"], g["offs"])
        torch.cuda.current_stream().wait_stream(s)
        g["graph"] = torch.cuda.CUDAGraph()
        # thread_local: work another thread issues meanwhile does not invalidate the capture
        with torch.cuda.graph(g["graph"], pool=self.pool, capture_error_mode="thread_local"):
            g["out"] = self._step(g["px"], g["mem"], g["mem_pos"], g["ptr"], g["offs"])
        self.graphs[key] = g
        return g

    def precapture(self, size: int) -> None:
        """Capture every graph up front on a blank frame, so a clip runs graphs from frame 1."""
        frame = torch.zeros((size, size, 3), dtype=torch.uint8, device="cuda")
        q = size / 4
        self.start(frame, [[q, q, 3 * q, 3 * q]])
        for _ in range(self.n_ptr):
            self(frame)
        torch.cuda.synchronize()
