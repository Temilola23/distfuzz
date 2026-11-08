"""Training profile of the DCP generator (`--profile train`): whole training steps drawn from production templates.

Scenarios keep the classic format (model, optim, segs with save/load), so run_segment, the reference and the
checkpoint oracles are shared. The profile adds: clip configs with every norm_type, foreach and the torchtitan /
accelerate clip variants; gradient accumulation with or without no_sync; activation checkpointing; LoRA and MoE
blocks; expert parallelism ("ep"); a loss-parallel head; split and model-only broadcast loads; and layer sizes that
are small or uneven relative to the world size.
"""

import copy

from distfuzz.dcp.gen import Gen, divisors
from distfuzz.dcp.runtime import model_dims_ok
from distfuzz.dcp.templates import TEMPLATES

TRAIN_PARS = ["ddp", "fsdp", "tp", "fsdp_tp", "hsdp", "ep"]
NORMS = [2.0, 1.0, "inf", 3.0]
SIZES = dict(d=[4, 6, 8, 12], h=[2, 3, 5, 6, 8, 12], vocab=[3, 5, 7, 8, 11, 16], out=[1, 2, 3, 5, 7],
             E=[2, 3, 4, 5, 8], eh=[2, 3, 4, 6], r=[1, 2, 3, 4, 8])  # fmt: skip


def norm_val(n):
    return "inf" if n == float("inf") or n == "inf" else float(n)


class TrainGen(Gen):
    def __init__(self, rng, max_w=4):
        super().__init__(rng, max_w=max_w)
        self.names = sorted(TEMPLATES)

    # model --------------------------------------------------------------------------------------------------
    def attn(self, d):
        return dict(t="attn", nh=self.r.choice([n for n in (1, 2, 3, 4) if d % n == 0]), bias=False)

    def moe(self):
        r = self.r
        E = r.choice(SIZES["E"])
        return dict(
            t="moe",
            E=E,
            topk=r.choice([1, 1, 2]) if E > 1 else 1,
            eh=r.choice(SIZES["eh"]),
            score=r.choice(["softmax", "sigmoid"]),
            route_norm=r.random() < 0.5,
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
        d = r.choice(SIZES["d"])
        m = dict(kind=kind, seq=r.choice([2, 3, 4]), d0=d, head_bias=r.random() < 0.3)
        if kind == "tok":
            m["vocab"] = r.choice(SIZES["vocab"])
        rms = r.random() < 0.7
        ln = dict(t="ln", affine=True, bias=False, rms=rms)
        blocks = []
        for _ in range(r.choice([1, 1, 2])):
            if lay == "llama":
                blocks += [dict(ln), self.attn(d), dict(ln), dict(t="mlp", h=r.choice(SIZES["h"]), bias=False,
                                                               act="gelu")]  # fmt: skip
            elif lay == "moe":
                blocks += [dict(ln), self.attn(d), dict(ln), self.moe()] if kind == "tok" else [self.moe()]
            elif lay == "lora":
                blocks += [dict(ln), self.attn(d), self.lora()] if kind == "tok" and r.random() < 0.5 else [self.lora()]
            elif lay == "mlp":
                blocks.append(
                    r.choice(
                        [
                            dict(t="mlp", h=r.choice(SIZES["h"]), bias=r.random() < 0.6, act="gelu"),
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
        if seg["w"] == 1:
            pars = ["none"]
        par = r.choice(pars)
        if par == "ep" and any(b["t"] == "moe" and b["E"] % seg["w"] for b in m["blocks"]):
            par = "fsdp"  # torchtitan requires E divisible by the EP degree (see valid())
        seg["par"] = par
        for k in ("tp", "units", "raf", "raf_root", "mp", "mesh2d", "ddp_bcast", "ddp_gabv", "a2a"):
            seg.pop(k, None)
        w = seg["w"]
        tpdeg = None
        if par in ("fsdp_tp", "hsdp"):
            a = r.choice([x for x in divisors(w) if 1 < x < w] or divisors(w))
            seg["mesh2d"] = [a, w // a]
            tpdeg = w // a
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
        if par in ("fsdp", "fsdp_tp", "hsdp", "ep"):
            units = [str(i) for i, b in enumerate(m["blocks"]) if b["t"] != "ln" or r.random() < 0.3]
            if not m.get("tie"):
                if m["kind"] == "tok" and r.random() < 0.7:
                    units.append("emb")
                if r.random() < 0.7:
                    units.append("head")
            seg["units"] = units
            seg["raf"] = r.choice([True, True, False])
            seg["raf_root"] = r.choice([True, False])
            if par != "ep" and r.random() < tp["mp"]:
                seg["mp"] = dict(param="bf16", reduce=r.choice(["f32", None]))
        if par == "ep":
            seg["a2a"] = r.choice(["functional", "c10d"])
        if par == "ddp":
            seg["ddp_bcast"] = True
            seg["ddp_gabv"] = r.random() < 0.3

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
        if r.random() < 0.25:
            return sv  # classic breadth (raw API, planner knobs, process async, safetensors)
        how = r.choice(tp["save"])
        sv["api"] = "stateful" if how == "stateful" else "sd"
        sv["mode"] = "sharded" if how in ("stateful", "sd") else how
        sv["ign_frozen"] = False
        sv["async"] = "thread" if r.random() < tp["async_"] else "none"
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
        if r.random() < 0.25:
            return Gen.gen_load(self, scn, seg, prev_sv)
        ld = Gen.gen_load(self, scn, seg, prev_sv)
        if ld["api"] != "stateful":
            ld["api"] = "sd"
            ld["mode"] = r.choice(tp["load"])
        return ld

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
            scn = dict(
                profile="train", tmpl=name, seed=r.randint(1, 10**6), dtype="f64" if r.random() < 0.85 else "f32"
            )
            scn["model"] = self.gen_model_t(tp)
            scn["optim"] = self.gen_optim_t(scn)
            self.gen_train_knobs(scn)
            scn["batch"] = 24
            scn["zg_none"] = True
            scn["flat_osd"] = False  # flatten_optimizer_state_dict + full_state_dict is the known A2 / VERIFIED §10
            scn["segs"] = [self.gen_seg(scn) for _ in range(r.choice([1, 2, 2, 3]))]
            self.fix_chain(scn)
            if self.valid(scn):
                return scn
        raise RuntimeError("could not generate a valid training scenario")

    def valid(self, scn):
        if not Gen.valid(self, scn):
            return False
        m = scn["model"]
        if not model_dims_ok(m):
            return False
        has_moe = any(b["t"] == "moe" for b in m["blocks"])
        has_lora = any(b["t"] == "lora" for b in m["blocks"])
        if any(b["t"] == "moe" and b["topk"] > b["E"] for b in m["blocks"]):
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
