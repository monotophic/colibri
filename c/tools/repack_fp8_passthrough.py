"""fmt=100 repack tool: Z.ai GLM-5.2-FP8 shards -> the engine's native-FP8-passthrough
container (byte-preserved, no dequant/requant).

fmt=100, not fmt=6: this tool's output format was minted fmt=6 during this
branch's original development, before dev's own #465 (E8/IQ3) claimed that
ordinal upstream and merged it into dev as a REAL fmt=6 (colibri.c's
qt_resolve_fmt, quant.h's E8 constants). Ported onto dev post-#465-merge as
fmt=100 from the start (PRIVATE ORDINAL BLOCK, see colibri.c's QT struct
comment) -- this tool has never emitted a container claiming ordinal 6.

Unlike convert_fp8_to_int4.py (which DEQUANTIZES fp8 -> f32 -> REQUANTIZES to a
different, lossy format), this tool copies the fp8 weight bytes AS-IS and only
renames/reshapes the scale sidecar to the engine's `.qs` convention. Same 8
bits/weight streaming cost as the source, zero additional quantization loss.

THE DESIGN LANDMINE (see qt_resolve_fmt in colibri.c, c/quant.h's E4M3_LUT
comment): the engine tells fmt=100 (this tool's output) apart from fmt=1 (plain
int8) ONLY by scale-array geometry -- per-row for fmt=1, per-128x128-block for
fmt=100. This tool must therefore emit .qs sidecars whose byte count is EXACTLY
ceil(O/128)*ceil(I/128)*4, never anything that could coincide with O*4 by
accident for the shapes it targets (see _check_geometry below, which refuses
rather than silently emitting a container the engine might misread). At I=98
specifically, a single-block fp8 tensor's raw weight-byte count ALSO coincides
with a genuine fmt=6 (E8/IQ3) tensor's -- see qt_resolve_fmt's reconciliation
comment and the build report's "THE SEMANTIC RECONCILIATION" section; this
tool does not target I=98 shapes in practice (Z.ai's real projections don't
land there), but the engine's read-side refusal covers it regardless.

SELECTED kinds only: "resident" tensors (shared expert, o_proj, other
attention projections, dense-MLP first layers, and the generic resident
fallback) -- routed experts (kind "x" in convert_fp8_to_int4.classify) are
EXCLUDED and stay on the existing int4-g64 streaming path; routers/norms/bias
(kind "f32") and embed/lm_head (kind "io", BF16 in source, never FP8) are
untouched by this tool -- carrying those into a mixed fmt=100+int4 loadout
directory is separate, deferred "loadout index-rewrite blending" work.

kv_b_proj (kind "kvb") is ALSO deliberately excluded, despite being a resident
tensor classify() would otherwise select: colibri.c's MLA-absorption CPU path
(qt_addrow/qt_matvec_rows, called only on l->kv_b -- the always-available
fallback whenever the Metal fused decode kernel isn't used) has no fmt==100
case and would silently misread fp8 bytes as int2-packed data; the CUDA
absorb kernels (coli_cuda_attention_absorb/_kvdev in backend_cuda.cu) are
similarly int4-specific. Repacking kv_b_proj to fmt=100 needs that CPU+CUDA
absorb-path work first -- out of scope for this build, so this tool refuses
to produce a container the engine cannot safely read.

HARD CONSTRAINT for this build: unit-tested on synthetic fixtures ONLY (see
tests/test_fp8_repack.py, built with tools/glm_fp8_emit.py's exact real-
checkpoint layout) plus --dry-run inventory mode against those same fixtures.
NO read of any real Z.ai shard, NO full or partial repack of a real checkpoint
-- that is later, user-GO'd work once this build's plumbing has been reviewed.

METADATA STAMP (reference implementation of the FORMATS-registry FR -- see
upstream_contribution/FORMATS_registry_draft.md): every output shard's
safetensors `__metadata__` carries a `colibri.fmt` key whose VALUE is itself a
JSON-encoded object mapping each stamped tensor's exact name to its format
NAME string (FORMAT_NAME below, "fp8-e4m3-b128" -- never the private ordinal
100, which is an internal enum value the container must never depend on, see
colibri.c's PRIVATE ORDINAL BLOCK comment). The reader (qt_from_disk's
qt_verify_fmt_stamp, colibri.c) is TRUST-VERIFY-REFUSE: a stamp that agrees
with the byte-arithmetic inference is a no-op (the tensor loads exactly as it
would unstamped); a stamp that disagrees, or names a format this build
doesn't recognize, is refused loudly (same "untrusted container" discipline
as qt_resolve_fmt's own THE DESIGN LANDMINE refusal). Unstamped containers
(everything this tool produced before this stamp existed, and any container
from a tool that doesn't stamp) are unaffected: no stamp means inference
alone decides, exactly as before this feature existed.

Usage (synthetic fixtures / local testing only):
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --dry-run
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --n-layers 78
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import classify, check_or_record_params  # reuse: same taxonomy,
                                                                    # same #383-class params guard

# Kinds classify() can return that this tool targets: resident weights, NOT routed
# experts ("x"), NOT the always-F32 set ("f32"), NOT embed/lm_head ("io" -- BF16 in
# source, never FP8), NOT sidecars/skips ("consumed"/"skip"). "kvb" (kv_b_proj) is
# ALSO excluded -- see the module docstring: the CPU absorb path (qt_addrow/
# qt_matvec_rows) and the CUDA absorb kernels have no fmt=100 case yet.
RESIDENT_KINDS = frozenset({"sh", "o", "attn", "dmlp", "q"})

# The format's PUBLIC identity (what containers/FRs advertise) -- never the
# private 100 ordinal, which is an internal-only enum value (see colibri.c's
# QT struct comment, PRIVATE ORDINAL BLOCK). Must match the reader's
# FMT_NAMES table (colibri.c, near qt_resolve_fmt) or every stamp this tool
# writes will be refused as "unrecognized format name" on load.
FORMAT_NAME = "fp8-e4m3-b128"

# safetensors __metadata__ key this tool stamps: JSON-encoded {tensor_name:
# format_name}. Kept as a module constant so the reader-side doc and any
# future tooling can cite the exact string instead of re-typing it.
METADATA_KEY = "colibri.fmt"

BLOCK = 128


def _nblk(n):
    return (n + BLOCK - 1) // BLOCK


def is_repack_target(name, dtype, keys, n_layers):
    """True if `name` is a resident-kind FP8 tensor this tool should byte-preserve
    into the fmt=100 container. `keys` is the full set of tensor names in the shard
    (needed to confirm the `_scale_inv` sidecar is actually present -- classify()
    alone can't tell FP8 tensors from any other dtype a resident-kind name might
    carry in a non-FP8 checkpoint variant)."""
    if name.endswith("_scale_inv"):
        return False                       # sidecar, handled together with its weight
    kind = classify(name, n_layers)
    if kind not in RESIDENT_KINDS:
        return False
    if dtype not in ("F8_E4M3", "float8_e4m3fn"):
        return False
    return (name + "_scale_inv") in keys


def _check_geometry(name, O, I, nblkO, nblkI):
    """Refuse (loud, ValueError) rather than silently emit a .qs whose geometry
    doesn't match what qt_resolve_fmt expects for [O,I] -- the same "untrusted
    container" discipline the C side applies on read, applied here on write so a
    malformed source checkpoint is caught at repack time, not at load time three
    steps later with a confusing engine-side refusal."""
    exp_nblkO, exp_nblkI = _nblk(O), _nblk(I)
    if (nblkO, nblkI) != (exp_nblkO, exp_nblkI):
        raise ValueError(
            f"{name}: scale_inv block shape ({nblkO},{nblkI}) != expected "
            f"({exp_nblkO},{exp_nblkI}) for [{O},{I}] -- refusing to repack a shape "
            f"the engine's qt_resolve_fmt would either refuse or (worse) misread")


def shard_inventory(path, n_layers):
    """--dry-run: scan one shard's header only (get_slice, no tensor data pulled)
    and return the list of tensors that WOULD be repacked, without writing
    anything. Cheap: safe to run over an entire real checkpoint's headers."""
    from safetensors import safe_open
    inv = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            if not is_repack_target(name, dt, keys, n_layers):
                continue
            O, I = sl.get_shape()
            sc_slice = f.get_slice(name + "_scale_inv")
            nblkO, nblkI = sc_slice.get_shape()
            _check_geometry(name, O, I, nblkO, nblkI)
            inv.append({"name": name, "kind": classify(name, n_layers),
                       "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                       "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
    return inv


def repack_shard(path, n_layers):
    """Real mode: byte-preserve every selected tensor's weight bytes (raw e4m3,
    no dequant/requant -- `.view(torch.uint8)` is a pure bit-reinterpret, verified
    byte-identical against torch.float8_e4m3fn's own storage) and rename/flatten
    the `_scale_inv` sidecar to the engine's `name.qs` convention (flat F32,
    nblkO*nblkI elements, row-major [blkO,blkI] -- matching
    scale[(o/128)*nblkI+i/128] on the read side). Returns (out_dict, inventory,
    fmt_map): fmt_map is {weight_tensor_name: FORMAT_NAME} for every selected
    tensor -- the caller JSON-encodes it into the output shard's __metadata__
    (see the module docstring's METADATA STAMP section). Keyed by the WEIGHT
    name only (not the ".qs" sidecar) -- that's the name qt_resolve_fmt/
    qt_verify_fmt_stamp look the stamp up by on the read side."""
    import torch
    from safetensors import safe_open
    out = {}
    inv = []
    fmt_map = {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            if not is_repack_target(name, dt, keys, n_layers):
                continue
            w = f.get_tensor(name)                       # torch.float8_e4m3fn [O,I]
            sc = f.get_tensor(name + "_scale_inv")        # torch.float32 [nblkO,nblkI]
            O, I = w.shape
            nblkO, nblkI = sc.shape
            _check_geometry(name, O, I, nblkO, nblkI)
            out[name] = w.view(torch.uint8).contiguous()              # BYTE-PRESERVED
            out[name + ".qs"] = sc.to(torch.float32).reshape(-1).contiguous()
            fmt_map[name] = FORMAT_NAME
            inv.append({"name": name, "kind": classify(name, n_layers),
                       "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                       "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
    return out, inv, fmt_map


def _print_inventory_summary(all_inv, dry_run):
    by_kind = {}
    tot_w = tot_s = 0
    for it in all_inv:
        by_kind.setdefault(it["kind"], []).append(it)
        tot_w += it["weight_bytes"]; tot_s += it["scale_bytes"]
    tag = "[DRY-RUN]" if dry_run else "[REPACK]"
    print(f"{tag} {len(all_inv)} tensor(s) selected across all shards "
         f"({tot_w/1e9:.3f} GB weights, {tot_s/1e6:.3f} MB scales)")
    for kind in sorted(by_kind):
        items = by_kind[kind]
        w = sum(it["weight_bytes"] for it in items)
        print(f"{tag}   kind={kind:5s} {len(items):5d} tensor(s)  {w/1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="directory of Z.ai FP8 *.safetensors shards")
    ap.add_argument("--outdir", required=True, help="destination for repacked fmt=100 container shards")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: scan headers, print selection counts, write nothing")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
    if not shards:
        print(f"ERROR: no *.safetensors files in {a.indir}"); sys.exit(1)

    if a.dry_run:
        all_inv = []
        for sp in shards:
            all_inv.extend(shard_inventory(sp, a.n_layers))
        _print_inventory_summary(all_inv, dry_run=True)
        return

    os.makedirs(a.outdir, exist_ok=True)
    # #383-class guard, reused (not reimplemented): refuses a resumed run whose
    # params differ from what's already partially in this outdir (the #355 failure
    # mode -- a second pass with different flags silently mixing containers).
    # Owns its own sidecar (.fp8pass-params.json), separate from the shard-progress
    # manifest below, exactly like the --repo mtp/indexer loops in
    # convert_fp8_to_int4.py use it.
    params = {"mode": "fp8-passthrough", "n_layers": a.n_layers}
    if not check_or_record_params(a.outdir, "fp8pass-", params):
        sys.exit(1)

    # RESUME (same #383-class manifest idiom as convert_fp8_to_int4.py's --indir path):
    # a sidecar records input-shard -> output-shard-name-or-"" (shards with no
    # resident-FP8 tensors emit no file), written atomically so an interrupted run
    # never leaves a half-written manifest for the next invocation to trust. The
    # params guard above already owns "don't mix conversions" -- this manifest is
    # purely which shards are done.
    prog_path = os.path.join(a.outdir, ".fp8pass-progress.json")
    prog = {}
    if os.path.exists(prog_path):
        try:
            prog = json.loads(open(prog_path).read())
        except (OSError, ValueError):
            prog = {}
    done = prog.setdefault("shards", {})

    from safetensors.torch import save_file
    all_inv = []
    n = fresh = skipped = 0
    for sp in shards:
        key = os.path.basename(sp)
        prev = done.get(key)
        if prev is not None and (prev == "" or os.path.exists(os.path.join(a.outdir, prev))):
            if prev:
                n += 1
            skipped += 1
            continue
        out, inv, fmt_map = repack_shard(sp, a.n_layers)
        all_inv.extend(inv)
        if not out:
            done[key] = ""
        else:
            name = f"out-fp8pass-{n:05d}.safetensors"
            # METADATA STAMP: __metadata__ values must be strings (safetensors
            # constraint) -- colibri.fmt's VALUE is itself JSON text (a map of
            # tensor name -> format NAME), not a nested object, for exactly
            # that reason. sort_keys for a deterministic, diffable stamp.
            meta = {METADATA_KEY: json.dumps(fmt_map, sort_keys=True)}
            save_file(out, os.path.join(a.outdir, name), metadata=meta)
            done[key] = name
            n += 1
            fresh += 1
        tmp = prog_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(prog, fh, indent=1)
        os.replace(tmp, prog_path)
    if skipped:
        print(f"[RESUME] {skipped} shard(s) already done in {a.outdir}, skipped")
    print(f"[REPACK] {fresh} new output shard(s) written")
    _print_inventory_summary(all_inv, dry_run=False)


if __name__ == "__main__":
    main()
