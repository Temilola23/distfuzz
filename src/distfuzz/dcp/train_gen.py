"""Training profile of the DCP generator (`--profile train`): whole training steps drawn from production templates.

Scenarios keep the classic format (model, optim, segs with save/load), so run_segment, the reference and the
checkpoint oracles are shared. The profile adds: clip configs with every norm_type, foreach and the torchtitan /
accelerate clip variants; gradient accumulation with or without no_sync; activation checkpointing; LoRA and MoE
blocks; expert parallelism ("ep"); a loss-parallel head; split and model-only broadcast loads; and layer sizes that
are small or uneven relative to the world size.
"""

import copy
import shutil

from distfuzz.dcp.gen import Gen, divisors
from distfuzz.dcp.runtime import model_dims_ok
from distfuzz.dcp.templates import TEMPLATES

TRAIN_PARS = ["ddp", "fsdp", "tp", "fsdp_tp", "hsdp", "ep", "pp", "pp_fsdp"]
PP_SCHEDS = ["GPipe", "1F1B", "Interleaved1F1B", "LoopedBFS", "InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV"]
MULTI = ("Interleaved1F1B", "LoopedBFS", "InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV")
try:  # float8 training needs torchao (the distfuzz:cpu-ao image)
    import importlib.util

    HAVE_AO = importlib.util.find_spec("torchao") is not None
except Exception:  # noqa: BLE001
    HAVE_AO = False
# inductor needs a C++ compiler (the distfuzz:cpu-cc image); aot_eager runs anywhere
BACKENDS = ["aot_eager"] + (["inductor"] if shutil.which("g++") or shutil.which("c++") else [])
NORMS = [2.0, 1.0, "inf", 3.0]
SIZES = dict(d=[4, 6, 8, 12], h=[2, 3, 5, 6, 8, 12], vocab=[3, 5, 7, 8, 11, 16], out=[1, 2, 3, 5, 7],
             E=[2, 3, 4, 5, 8], eh=[2, 3, 4, 6], r=[1, 2, 3, 4, 8])  # fmt: skip


def norm_val(n):
    return "inf" if n == float("inf") or n == "inf" else float(n)


SEG_KEYS = ("tp", "units", "raf", "raf_root", "mp", "mesh2d", "ddp_bcast", "ddp_gabv", "a2a", "sp", "pp", "ppmesh",
            "fsdp_knobs", "groups", "mesh_from_group")  # fmt: skip


def pp_units_of(scn):
    m = scn["model"]
    return (["emb"] if m["kind"] == "tok" else []) + [str(i) for i in range(len(m["blocks"]))] + ["head"]


def is_pp(seg):
    return seg.get("par") in ("pp", "pp_fsdp")


class TrainGen(Gen):
    def __init__(self, rng, max_w=4):
        super().__init__(rng, max_w=max_w)
        self.names = sorted(TEMPLATES)

    # model --------------------------------------------------------------------------------------------------
    def attn(self, d):
        nhs = [n for n in (1, 2, 3, 4) if d % n == 0]
        if getattr(self, "_even", False):
            nhs = [n for n in nhs if n % 2 == 0] or nhs
        return dict(t="attn", nh=self.r.choice(nhs), bias=False)

    def moe(self):
        r = self.r
        E = r.choice(SIZES["E"])
        topk = r.choice([1, 1, 2]) if E > 1 else 1
        return dict(
            t="moe",
            E=E,
            topk=topk,
            eh=r.choice(SIZES["eh"]),
            score=r.choice(["softmax", "sigmoid"]),
            # with top-1 routing, route_norm makes every routing weight 1: the router gets an analytically zero
            # gradient, and Adam turns its rounding noise into lr-sized steps (a reference false positive)
            route_norm=topk > 1 and r.random() < 0.5,
            shared=r.random() < 0.3,
        )

    def lora(self):
        r = self.r
        rank = r.choice(SIZES["r"])
        return dict(t="lora", h=r.choice(SIZES["h"]), r=rank, alpha=2 * rank, bias=r.random() < 0.3)

    def gen_model_t(self, tp):
        r = self.r
        kind = tp["kind"] if tp["kind"] != "any" else r.choice(["tok", "vec"])
        lay = tp["layout"] if tp["layout"] != "any" else r.choice(["llama", "mlp", "lora", "moe", "embhead"])
        # TP-friendly sizes (multiples of 4) for templates that run sequence parallelism, most of the time
        self._even = r.random() < 0.7 and tp.get("sp", 0) > 0
        big = tp.get("big", False)  # float8 layers need dims divisible by 16
        d = r.choice([16, 32] if big else [4, 8, 12] if self._even else SIZES["d"])
        m = dict(kind=kind, seq=r.choice([4, 8] if self._even else [2, 3, 4]), d0=d, head_bias=r.random() < 0.3)
        if kind == "tok":
            m["vocab"] = r.choice([16, 32] if big else [8, 16] if self._even else SIZES["vocab"])
        rms = r.random() < 0.7
        ln = dict(t="ln", affine=True, bias=False, rms=rms)
        blocks = []
        for _ in range(r.choice([1, 1, 2])):
            if lay == "llama":
                h = r.choice([16, 32, 48] if big else [4, 8, 12] if self._even else SIZES["h"])
                blocks += [dict(ln), self.attn(d), dict(ln), dict(t="mlp", h=h, bias=False, act="gelu")]
            elif lay == "moe":
                blocks += [dict(ln), self.attn(d), dict(ln), self.moe()] if kind == "tok" else [self.moe()]
            elif lay == "lora":
                blocks += [dict(ln), self.attn(d), self.lora()] if kind == "tok" and r.random() < 0.5 else [self.lora()]
            elif lay == "mlp":
                blocks.append(
                    r.choice(
                        [
                            dict(
                                t="mlp",
                                h=r.choice([16, 32, 48] if big else SIZES["h"]),
                                bias=r.random() < 0.6,
                                act="gelu",
                            ),  # fmt: skip
                            dict(t="lin", out=d, bias=r.random() < 0.6, act=r.choice(["none", "relu", "tanh"])),
                        ]
                    )
                )
            elif lay == "embhead":
                blocks = [dict(ln)] if r.random() < 0.5 else []
                break
        m["blocks"] = blocks
        m["tie"] = kind == "tok" and lay == "embhead" and r.random() < 0.3
        m["out"] = m["vocab"] if kind == "tok" else r.choice(SIZES["out"])
        m["frozen"] = []
        m["lora_only"] = lay == "lora" and r.random() < 0.7  # torchtune trains adapters only
        return m

    # parallelism --------------------------------------------------------------------------------------------
    def gen_par(self, scn, seg):
        r = self.r
        tp = TEMPLATES[scn["tmpl"]]
        m = scn["model"]
        pars = list(tp["pars"]) if r.random() < 0.8 else list(TRAIN_PARS)
        if not any(b["t"] == "moe" for b in m["blocks"]):
            pars = [p for p in pars if p != "ep"] or ["fsdp"]
        if any(b["t"] in ("moe", "lora") for b in m["blocks"]):
            pars = [p for p in pars if p not in ("tp", "fsdp_tp")] or ["fsdp"]
        if m.get("tie") or m.get("lora_only"):
            pars = [p for p in pars if p not in ("pp", "pp_fsdp")] or ["fsdp"]
        if seg["w"] != 4:
            pars = [p for p in pars if p != "pp_fsdp"] or (["pp"] if "pp" in pars else ["fsdp"])
        if seg["w"] == 1:
            pars = ["none"]
        par = r.choice(pars)
        if par == "ep" and any(b["t"] == "moe" and b["E"] % seg["w"] for b in m["blocks"]):
            par = "fsdp"  # torchtitan requires E divisible by the EP degree (see valid())
        seg["par"] = par
        for k in SEG_KEYS:
            seg.pop(k, None)
        w = seg["w"]
        tpdeg = None
        if par in ("fsdp_tp", "hsdp"):
            # 2 x 2 mostly; 1 x 4 and 4 x 1 (a size-1 FSDP or TP dim) a quarter of the time
            mid = [x for x in divisors(w) if 1 < x < w]
            a = r.choice(mid if mid and r.random() < 0.75 else divisors(w))
            seg["mesh2d"] = [a, w // a]
            tpdeg = w // a
        if par in ("pp", "pp_fsdp"):
            self.gen_pp(scn, seg)
        if par == "tp":
            tpdeg = w
        if par in ("tp", "fsdp_tp"):
            plan = {}
            for i, b in enumerate(m["blocks"]):
                # production TP needs dims divisible by the degree (torchtitan asserts it); 10% stay uneven
                even = r.random() < 0.9
                if r.random() < 0.85:
                    if b["t"] == "lin":
                        st = r.choice(["col", "row"])
                        din = m["d0"]  # lin blocks keep d (gen_model_t uses out=d)
                        if not even or (b["out"] if st == "col" else din) % tpdeg == 0:
                            plan[str(i)] = st
                    elif b["t"] == "mlp" and (not even or b["h"] % tpdeg == 0):
                        plan[str(i)] = "mlp"
                    elif b["t"] == "attn" and b["nh"] % tpdeg == 0:
                        plan[str(i)] = "attn"
            if not m.get("tie"):
                if m["kind"] == "tok" and r.random() < 0.4:
                    plan["emb"] = "row"  # vocab-sharded embedding (torchtitan)
                if m["kind"] == "tok" and m["vocab"] % tpdeg and r.random() < 0.9:
                    plan.pop("emb", None)  # vocab < degree is VERIFIED §15; keep it rare
                if r.random() < 0.7 and (m["out"] % tpdeg == 0 or r.random() < 0.1):
                    # out < degree leaves an empty colwise shard: grad comes back Replicate (VERIFIED §9 family)
                    lp = m["kind"] == "tok" and r.random() < tp["lp"]
                    plan["head"] = "col_lp" if lp else "col"
            seg["tp"] = plan
            if r.random() < tp.get("sp", 0.0):
                self.make_sp(scn, seg, tpdeg)
        if par in ("fsdp", "fsdp_tp", "hsdp", "ep", "pp_fsdp"):
            units = [str(i) for i, b in enumerate(m["blocks"]) if b["t"] != "ln" or r.random() < 0.3]
            if not m.get("tie"):
                if m["kind"] == "tok" and r.random() < 0.7:
                    units.append("emb")
                if r.random() < 0.7:
                    units.append("head")
            seg["units"] = units
            fdeg = {"fsdp": w, "ep": w, "hsdp": seg.get("mesh2d", [1, w])[1], "fsdp_tp": seg.get("mesh2d", [w])[0],
                    "pp_fsdp": seg.get("ppmesh", [1, w])[1]}[par]  # fmt: skip
            ints = [k for k in (2,) if fdeg > k and fdeg % k == 0 and par in ("fsdp", "ep")]
            # Megatron-Core passes reshard_after_forward as bool or int (reshard to a smaller group)
            seg["raf"] = r.choice([True, True, False] + (ints if tp.get("mcore") else []))
            seg["raf_root"] = r.choice([True, False])
            if par not in ("ep", "pp_fsdp") and r.random() < tp["mp"]:
                seg["mp"] = dict(param="bf16", reduce=r.choice(["f32", None]))
            if r.random() < tp.get("knobs", 0.0):
                kn = dict(
                    gdf1=r.random() < 0.5,
                    emb_no_bwd_unshard="emb" in units and r.random() < 0.5,
                    fwd_prefetch=r.random() < 0.3,
                    bwd_prefetch=r.random() < 0.3,
                    defer_reduce=scn.get("accum", 1) > 1 and r.random() < 0.5,
                )
                seg["fsdp_knobs"] = {k: v for k, v in kn.items() if v}
                last = str(len(m["blocks"]) - 1)
                if "head" in units and m["blocks"] and m["blocks"][-1]["t"] == "ln" and r.random() < 0.5:
                    seg["groups"] = [[last, "head"]]  # torchtitan fully_shard([norm, lm_head])
            if tp.get("mcore"):
                seg["mesh_from_group"] = par == "fsdp"
                if scn.get("ac", "none") != "none":
                    seg["fsdp_knobs"] = dict(seg.get("fsdp_knobs") or {}, bwd_prefetch=True)
        if par == "ep":
            seg["a2a"] = r.choice(["functional", "c10d"])
        if par == "ddp":
            seg["ddp_bcast"] = True
            seg["ddp_gabv"] = r.random() < 0.3

    def make_sp(self, scn, seg, tpdeg):
        """Turn a TP plan into a sequence-parallel one (torchtitan's default): every block planned, norms
        SequenceParallel, embedding and head planned, the sequence divisible by the degree."""
        r = self.r
        m = scn["model"]
        if m["kind"] != "tok" or m.get("tie") or m["seq"] % tpdeg:
            return
        plan = {}
        for i, b in enumerate(m["blocks"]):
            if b["t"] == "ln":
                plan[str(i)] = "sp"
            elif b["t"] == "mlp" and b["h"] % tpdeg == 0:
                plan[str(i)] = "mlp"
            elif b["t"] == "attn" and b["nh"] % tpdeg == 0 and m["d0"] % tpdeg == 0:
                plan[str(i)] = "attn"
            elif b["t"] == "lin" and b["out"] % tpdeg == 0:
                plan[str(i)] = r.choice(["col", "row"]) if m["d0"] % tpdeg == 0 else "col"
            else:
                return
        plan["emb"] = r.choice(["row", "col"]) if m["vocab"] % tpdeg == 0 and m["d0"] % tpdeg == 0 else None
        if plan["emb"] is None:
            return
        if m["out"] % tpdeg:
            return
        lp = r.random() < TEMPLATES[scn["tmpl"]]["lp"]
        plan["head"] = "col_lp" if lp else "col"
        seg["tp"] = plan
        seg["sp"] = True

    def gen_pp(self, scn, seg):
        r = self.r
        w = seg["w"]
        if seg["par"] == "pp_fsdp":
            seg["ppmesh"] = [2, w // 2]
        else:
            seg["ppmesh"] = [w, 1]
        pp = seg["ppmesh"][0]
        n = len(pp_units_of(scn))
        scheds = [x for x in PP_SCHEDS if (x not in MULTI and n >= pp) or (x in MULTI and n >= 2 * pp)]
        sched = r.choice(scheds or ["GPipe"])
        # torch requires n_microbatches >= stages for 1F1B-style schedules; looped / V schedules want a multiple of pp
        nst = pp * (2 if sched in MULTI else 1)
        local = scn["batch"] // seg["ppmesh"][1]
        mbs = [k for k in (1, 2, 3, 4, 6, 8, 12) if local % k == 0 and (sched == "GPipe" or k >= nst)]
        if sched in MULTI:
            mbs = [k for k in mbs if k % pp == 0] or mbs
        seg["pp"] = dict(sched=sched, mb=r.choice(mbs), vps=2 if sched in MULTI else 1,
                         titan_loss=r.random() < 0.5)  # fmt: skip

    # checkpoint ---------------------------------------------------------------------------------------------
    def gen_save(self, scn, seg, nxt):
        r = self.r
        tp = TEMPLATES[scn["tmpl"]]
        sv = Gen.gen_save(self, scn, seg, nxt)
        # flatten_state_dict=False is the known A1 / VERIFIED §4; the classic profile keeps covering it
        sv["planner"]["flat"] = True
        # FileSystemWriter's SAFETENSORS format is the HuggingFaceStorageWriter backend; FileSystemReader cannot
        # read it back (torch.load of a safetensors file), so the pairing is unsupported, not a finding
        sv["writer"]["fmt"] = "torch_save"
        pp = is_pp(seg) or is_pp(nxt)
        if r.random() < 0.25 and not pp:
            return sv  # classic breadth (raw API, planner knobs, process async, safetensors)
        how = "titan" if pp else r.choice(tp["save"])
        sv["api"] = how if how in ("stateful", "titan", "raw") else "sd"
        sv["mode"] = "sharded" if how in ("stateful", "sd", "titan", "raw") else how
        sv["ign_frozen"] = False
        sv["async"] = "thread" if r.random() < tp["async_"] and not pp else "none"
        if sv["async"] == "none":
            sv.pop("mutate", None)
            sv.pop("async_pg", None)
        else:
            sv["mutate"] = r.choice([0, 1])
            sv["async_pg"] = "new"
        sv["planner"] = dict(flat=True, flat_sh=True, dedup_low=False, cache=False)
        sv["writer"]["fmt"] = "torch_save"
        sv["presave"] = False
        return sv

    def gen_load(self, scn, seg, prev_sv):
        r = self.r
        tp = TEMPLATES[scn["tmpl"]]
        if prev_sv.get("api") == "titan":
            return dict(api="titan", mode="sharded", bcast_offload=False, planner=dict(flat=True, flat_sh=True))
        if r.random() < 0.25:
            return Gen.gen_load(self, scn, seg, prev_sv)
        ld = Gen.gen_load(self, scn, seg, prev_sv)
        if ld["api"] == "raw" and tp.get("mcore"):
            ld["legacy"] = r.random() < 0.5
        elif ld["api"] not in ("stateful", "titan"):
            ld["api"] = "sd"
            ld["mode"] = r.choice(tp["load"])
        return ld

    @staticmethod
    def load_ok(seg, sv):
        if not Gen.load_ok(seg, sv):
            return False
        return (sv["api"] == "titan") == (seg["load"]["api"] == "titan")

    def gen_seg(self, scn):
        r = self.r
        seg = dict(w=r.randint(2, self.max_w), steps=r.choice([1, 1, 2, 2, 3]))
        self.gen_par(scn, seg)
        seg["resave"] = r.random() < 0.2
        return seg

    # scenario -----------------------------------------------------------------------------------------------
    def gen_train_knobs(self, scn):
        r = self.r
        tp = TEMPLATES[scn["tmpl"]]
        c = tp["clip"]
        mx = r.choice(c["max"])
        if r.random() < 0.15:  # stacks expose norm_type; sample all of them now and then
            norm = r.choice(NORMS)
        else:
            norm = r.choice(c["norm"])
        scn["clip"] = (
            None
            if mx is None
            else dict(max=mx, norm=norm_val(norm), foreach=r.choice([None, True, False]), impl=r.choice(c["impl"]))
        )
        scn["accum"] = r.choice(tp["accum"])
        scn["no_sync"] = bool(tp["no_sync"]) and scn["accum"] > 1 and r.random() < 0.7
        scn["ac"] = r.choice(tp["ac"])
        scn["named_opt"] = r.random() < tp.get("named_opt", 0.0)
        scn["f8"] = None
        if HAVE_AO and r.random() < tp.get("f8", 0.0):
            scn["f8"] = dict(ag=r.random() < 0.7, precompute=r.random() < 0.7)
        scn["compile"] = None
        if r.random() < tp.get("compile", 0.0):
            # fullgraph is left off: FSDP2 hooks and checkpoint_wrapper are dynamo-disabled, so a fullgraph block
            # compile under them is a configuration error, not a finding
            scn["compile"] = dict(
                backend=r.choice(BACKENDS),
                how=r.choice(tp.get("compile_how", ["wrap", "inplace"])),
                fullgraph=False,
            )

    def gen_optim_t(self, scn):
        r = self.r
        tp = TEMPLATES[scn["tmpl"]]
        if r.random() < 0.2:
            return self.gen_optim()
        o = dict(tp["optim"])
        o["foreach"] = None
        o["fused"] = r.choice(tp["fused"]) is True
        if o["name"] in ("adam", "adamw"):
            o["amsgrad"] = False
        if o["name"] == "sgd":
            o.update(momentum=0.9, nesterov=False)
        return o

    def generate(self):
        r = self.r
        for _ in range(200):
            name = r.choice(self.names)
            tp = TEMPLATES[name]
            dt = "f32" if tp.get("f8") else "f64" if r.random() < 0.85 else "f32"
            scn = dict(profile="train", tmpl=name, seed=r.randint(1, 10**6), dtype=dt)
            scn["model"] = self.gen_model_t(tp)
            scn["optim"] = self.gen_optim_t(scn)
            self.gen_train_knobs(scn)
            scn["batch"] = 24
            scn["zg_none"] = True
            scn["flat_osd"] = False  # flatten_optimizer_state_dict + full_state_dict is the known A2 / VERIFIED §10
            scn["segs"] = [self.gen_seg(scn) for _ in range(r.choice([1, 2, 2, 3]))]
            self.fix_chain(scn)
            self.finish(scn)
            if self.valid(scn):
                return scn
        raise RuntimeError("could not generate a valid training scenario")

    def finish(self, scn):
        # torchtitan's checkpointer needs optimizers built from named parameters (param_names)
        if any(sg.get("save", {}).get("api") == "titan" for sg in scn["segs"]):
            scn["named_opt"] = True

    def valid(self, scn):
        if not Gen.valid(self, scn):
            return False
        if not self.valid_v2(scn):
            return False
        m = scn["model"]
        if not model_dims_ok(m):
            return False
        has_moe = any(b["t"] == "moe" for b in m["blocks"])
        has_lora = any(b["t"] == "lora" for b in m["blocks"])
        if any(b["t"] == "moe" and (b["topk"] > b["E"] or (b["topk"] == 1 and b["route_norm"])) for b in m["blocks"]):
            return False
        for s in scn["segs"]:
            par = s["par"]
            if par == "ep" and (not has_moe or s.get("mp")):
                return False
            # a rank with no experts skips the expert matmuls, so its autograd order differs and its FSDP
            # reduce-scatter can run before the combine all_to_all's backward the others wait in (cross-group
            # deadlock); torchtitan rules this out by requiring E % ep == 0
            if par == "ep" and any(b["t"] == "moe" and b["E"] % s["w"] for b in m["blocks"]):
                return False
            if par in ("tp", "fsdp_tp") and (has_moe or has_lora):
                return False
            dp = {"tp": 1, "none": 1, "fsdp_tp": s.get("mesh2d", [1])[0]}.get(par, s["w"])
            if scn["batch"] % (dp * scn.get("accum", 1)):
                return False
            if s.get("tp", {}).get("head") == "col_lp" and (m["kind"] != "tok" or m.get("tie")):
                return False
            if s.get("load", {}).get("mode") in ("full_bcast_split", "model_bcast") and s["load"]["api"] != "sd":
                return False
            # a resave after a model-only load runs get_state_dict's zero-lr optimizer init step on the fresh
            # optimizer: the known phantom step (#164929), which then shifts every later Adam step
            if s.get("load", {}).get("mode") == "model_bcast" and s.get("resave"):
                return False
            # replicate_leftovers distributes emb and head separately, which would untie a tied head
            if par in ("tp", "fsdp_tp") and m.get("tie"):
                return False
            # async_save in a thread on the default group races the training collectives (#123447, known);
            # torchtitan passes its own process group
            sv = s.get("save", {})
            if sv.get("async") == "thread" and sv.get("async_pg") != "new":
                return False
        return True

    def valid_v2(self, scn):
        m = scn["model"]
        cc = scn.get("clip")
        f8 = scn.get("f8")
        if f8:
            if scn["dtype"] != "f32" or (f8.get("precompute") and not f8.get("ag")):
                return False
            if any(s["par"] not in ("fsdp", "hsdp", "ddp") or s.get("mp") for s in scn["segs"]):
                return False
        for i, s in enumerate(scn["segs"]):
            par = s["par"]
            if s.get("sp"):
                if par not in ("tp", "fsdp_tp") or m["kind"] != "tok" or m.get("tie"):
                    return False
                k = s["w"] if par == "tp" else s["mesh2d"][1]
                if m["seq"] % k:  # uneven sequence shards: DTensor.from_local assumes even shards (known misuse)
                    return False
                plan = s.get("tp", {})
                if any(str(j) not in plan for j in range(len(m["blocks"]))) or "emb" not in plan or "head" not in plan:
                    return False
                if any(plan[str(j)] == "sp" and b["t"] != "ln" for j, b in enumerate(m["blocks"])):
                    return False
            elif any(v == "sp" for v in s.get("tp", {}).values()):
                return False
            if is_pp(s):
                pc = s.get("pp")
                if not pc or m.get("tie") or m.get("lora_only") or scn.get("accum", 1) != 1 or scn.get("no_sync"):
                    return False
                pp, dp = s.get("ppmesh", [0, 0])
                if pp * dp != s["w"] or (s["par"] == "pp" and dp != 1) or (s["par"] == "pp_fsdp" and dp < 2):
                    return False
                if len(pp_units_of(scn)) < pp * pc["vps"] or (scn["batch"] // dp) % pc["mb"]:
                    return False
                if pc["sched"] != "GPipe" and pc["mb"] < pp * pc["vps"]:
                    return False
                # selective AC caches each saved op once and refuses a second backward through the region
                # (torch/utils/checkpoint.py _CachedTorchDispatchMode); split-backward schedules need two
                if scn.get("ac") == "sac" and pc["sched"] in ("InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV"):
                    return False
                # torch rejects compiled stage modules under the split-backward (zero-bubble) schedules
                if scn.get("compile") and pc["sched"] in ("InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV"):
                    return False
                if isinstance(cc, dict) and cc.get("impl") not in ("titan", None):
                    return False
                if isinstance(cc, (int, float)) and cc:
                    return False
                if s.get("mp"):
                    return False
            elif s.get("pp"):
                return False
            sv = s.get("save", {})
            nxt = scn["segs"][i + 1] if i + 1 < len(scn["segs"]) else None
            pp_edge = is_pp(s) or (nxt is not None and is_pp(nxt))
            if sv and pp_edge and (sv["api"] != "titan" or sv["async"] != "none"):
                return False
            if sv.get("api") == "titan" and (sv.get("mode") != "sharded" or not scn.get("named_opt")):
                return False
            if s.get("mesh_from_group") and par != "fsdp":
                return False
            # the raw API saves Module.state_dict(), whose keys carry torch.compile's "_orig_mod." (and DDP's
            # "module.") prefixes; only the DSD / torchtitan paths canonicalize them, so raw + compile is a user error
            if scn.get("compile") and (sv.get("api") == "raw" or s.get("load", {}).get("api") == "raw"):
                return False
            if sv.get("api") == "raw" and (par == "ddp" or (nxt is not None and nxt["par"] == "ddp")):
                return False  # DDP's state_dict keys carry "module."
            if s.get("groups") and par not in ("fsdp", "fsdp_tp", "hsdp", "pp_fsdp"):
                return False
        return True

    def mutate(self, parent):
        r = self.r
        for _ in range(20):
            s = copy.deepcopy(parent)
            s.pop("id", None)
            k = r.randrange(6)
            segs = s["segs"]
            i = r.randrange(len(segs))
            if k == 0:
                s["model"] = self.gen_model_t(TEMPLATES[s["tmpl"]])
                for sg in segs:
                    self.gen_par(s, sg)
                    sg.pop("save", None)
                    sg.pop("load", None)
            elif k == 1:
                self.gen_train_knobs(s)
            elif k == 2:
                segs[i]["w"] = r.randint(2, self.max_w)
                self.gen_par(s, segs[i])
            elif k == 3:
                s["optim"] = self.gen_optim_t(s)
            elif k == 4 and len(segs) > 1:
                j = r.randrange(len(segs) - 1)
                segs[j]["save"] = self.gen_save(s, segs[j], segs[j + 1])
                segs[j + 1]["load"] = self.gen_load(s, segs[j + 1], segs[j]["save"])
            else:
                if len(segs) < 3 and r.random() < 0.6:
                    segs.insert(i + 1, self.gen_seg(s))
                elif len(segs) > 1:
                    del segs[i]
                for sg in segs:
                    sg.pop("load", None)
            self.fix_chain(s)
            self.finish(s)
            if self.valid(s):
                return s
        return self.generate()

    def simplifications(self, scn):
        out = [c for c in Gen.simplifications(self, scn) if self.valid(c)]

        def c(fn):
            s = copy.deepcopy(scn)
            s.pop("id", None)
            try:
                fn(s)
            except (KeyError, IndexError, ValueError):
                return
            self.fix_chain_min(s)
            if self.valid(s) and s != scn:
                out.append(s)

        if scn.get("accum", 1) > 1:
            c(lambda s: s.update(accum=1, no_sync=False))
        if scn.get("ac", "none") != "none":
            c(lambda s: s.__setitem__("ac", "none"))
        cl = scn.get("clip")
        if isinstance(cl, dict):
            if cl.get("impl", "torch") != "torch":
                c(lambda s: s["clip"].__setitem__("impl", "torch"))
            if cl.get("foreach") is not None:
                c(lambda s: s["clip"].__setitem__("foreach", None))
            if cl.get("max") != 1.0:
                c(lambda s: s["clip"].__setitem__("max", 1.0))
        if scn["model"].get("lora_only"):
            c(lambda s: s["model"].__setitem__("lora_only", False))
        if scn.get("compile"):
            c(lambda s: s.__setitem__("compile", None))
            if scn["compile"].get("fullgraph"):
                c(lambda s: s["compile"].__setitem__("fullgraph", False))
            if scn["compile"]["backend"] != "aot_eager":
                c(lambda s: s["compile"].__setitem__("backend", "aot_eager"))
        if scn.get("named_opt"):
            c(lambda s: s.__setitem__("named_opt", False))
        for i, sg in enumerate(scn["segs"]):
            if sg.get("sp"):

                def nosp(s, i=i):
                    g = s["segs"][i]
                    g["sp"] = False
                    g["tp"] = {k: v for k, v in g["tp"].items() if v != "sp"}

                c(nosp)
            for k in list(sg.get("fsdp_knobs") or {}):
                c(lambda s, i=i, k=k: s["segs"][i]["fsdp_knobs"].pop(k))
            if sg.get("groups"):
                c(lambda s, i=i: s["segs"][i].pop("groups"))
            if sg.get("mesh_from_group"):
                c(lambda s, i=i: s["segs"][i].__setitem__("mesh_from_group", False))
            if sg.get("pp"):
                for k, v in (("sched", "GPipe"), ("vps", 1), ("titan_loss", False), ("mb", 1)):
                    if sg["pp"].get(k) != v:
                        c(lambda s, i=i, k=k, v=v: s["segs"][i]["pp"].__setitem__(k, v))
            if "load" in sg and sg["load"].get("legacy"):
                c(lambda s, i=i: s["segs"][i]["load"].__setitem__("legacy", False))
        # shrink sizes (smaller layers keep empty-shard and uneven-split triggers alive)
        m = scn["model"]
        for key in ("d0", "vocab", "out", "seq"):
            if key in m:
                for v in sorted({1, 2, 3, 4, m[key] // 2}):
                    if 0 < v < m[key]:
                        c(lambda s, key=key, v=v: s["model"].__setitem__(key, v))
        for j, b in enumerate(m["blocks"]):
            for key in ("h", "E", "eh", "r", "topk", "nh"):
                if key in b:
                    for v in sorted({1, 2, b[key] // 2}):
                        if 0 < v < b[key]:
                            c(lambda s, j=j, key=key, v=v: s["model"]["blocks"][j].__setitem__(key, v))
            for key in ("shared", "route_norm", "bias"):
                if b.get(key):
                    c(lambda s, j=j, key=key: s["model"]["blocks"][j].__setitem__(key, False))
        return out
