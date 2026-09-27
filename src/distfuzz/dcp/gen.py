import copy
import json

from distfuzz.dcp.runtime import model_dims_ok

PARS = ["none", "ddp", "fsdp", "tp", "fsdp_tp", "hsdp"]


def divisors(n):
    return [a for a in range(1, n + 1) if n % a == 0]


class Gen:
    def __init__(self, rng, max_w=4):
        self.r = rng
        self.max_w = max_w

    def gen_model(self):
        r = self.r
        kind = r.choice(["tok", "vec"])
        m = dict(kind=kind, seq=r.choice([2, 3]), d0=r.choice([3, 4, 5, 6, 7, 8, 9, 12]), head_bias=r.random() < 0.6)
        if kind == "tok":
            m["vocab"] = r.choice([5, 7, 11, 13, 16])
        blocks = []
        d = m["d0"]
        for _ in range(r.randint(1, 4)):
            t = r.choice(["lin", "mlp", "mlp", "attn", "ln", "buf"])
            if t == "lin":
                b = dict(
                    t=t,
                    out=r.choice([3, 4, 5, 6, 7, 8, 9]),
                    bias=r.random() < 0.7,
                    act=r.choice(["none", "tanh", "relu", "gelu"]),
                )
                d = b["out"]
            elif t == "mlp":
                b = dict(
                    t=t,
                    h=r.choice([2, 3, 5, 7, 8, 11, 13]),
                    bias=r.random() < 0.7,
                    act=r.choice(["tanh", "relu", "gelu"]),
                )
            elif t == "attn":
                nhs = [n for n in (1, 2, 3, 4) if d % n == 0]
                b = dict(t=t, nh=r.choice(nhs), bias=r.random() < 0.5)
            elif t == "ln":
                aff = r.random() < 0.8
                b = dict(t=t, affine=aff, bias=aff and r.random() < 0.7)
            else:
                b = dict(t=t)
            blocks.append(b)
        m["blocks"] = blocks
        if kind == "tok" and d == m["d0"] and r.random() < 0.35:
            m["tie"] = True
            m["out"] = m["vocab"]
        else:
            m["out"] = m["vocab"] if kind == "tok" and r.random() < 0.6 else r.choice([2, 3, 5, 7])
        withp = [
            i for i, b in enumerate(blocks) if b["t"] in ("lin", "mlp", "attn") or (b["t"] == "ln" and b["affine"])
        ]
        m["frozen"] = [r.choice(withp)] if withp and r.random() < 0.2 else []
        return m

    def gen_optim(self):
        r = self.r
        name = r.choice(["sgd", "adam", "adamw"])
        o = dict(
            name=name,
            lr=r.choice([0.01, 0.03, 0.05]),
            wd=r.choice([0.0, 0.0, 0.01]),
            foreach=r.choice([None, True, False]),
            fused=r.random() < 0.1,
        )
        if name == "sgd":
            o["momentum"] = r.choice([0.0, 0.9, 0.9])
            o["nesterov"] = o["momentum"] > 0 and r.random() < 0.3
        else:
            o["amsgrad"] = r.random() < 0.25
        return o

    def gen_par(self, scn, seg):
        r = self.r
        w = seg["w"]
        m = scn["model"]
        par = r.choice(PARS if w == 1 else PARS[1:])
        seg["par"] = par
        for k in ("tp", "units", "raf", "raf_root", "mp", "mesh2d", "ddp_bcast", "ddp_gabv"):
            seg.pop(k, None)
        tpdeg = None
        if par in ("fsdp_tp", "hsdp"):
            a = r.choice(divisors(w))
            seg["mesh2d"] = [a, w // a]
            tpdeg = w // a
        if par == "tp":
            tpdeg = w
        if par in ("tp", "fsdp_tp"):
            plan = {}
            for i, b in enumerate(m["blocks"]):
                if r.random() < 0.6:
                    if b["t"] == "lin":
                        plan[str(i)] = r.choice(["col", "row"])
                    elif b["t"] == "mlp":
                        plan[str(i)] = "mlp"
                    elif b["t"] == "attn" and b["nh"] % tpdeg == 0:
                        plan[str(i)] = "attn"
            if not m.get("tie"):
                if m["kind"] == "tok" and r.random() < 0.5:
                    plan["emb"] = r.choice(["row", "col"])
                if r.random() < 0.5:
                    plan["head"] = r.choice(["col", "row"])
            seg["tp"] = plan
        if par in ("fsdp", "fsdp_tp", "hsdp"):
            units = [str(i) for i in range(len(m["blocks"])) if r.random() < 0.6]
            if not m.get("tie"):
                if m["kind"] == "tok" and r.random() < 0.3:
                    units.append("emb")
                if r.random() < 0.3:
                    units.append("head")
            seg["units"] = units
            seg["raf"] = r.choice([True, False, True, 2]) if par == "fsdp" and w == 4 else r.choice([True, False])
            seg["raf_root"] = r.choice([True, False])
            if r.random() < 0.2:
                seg["mp"] = dict(param=r.choice(["f32", "bf16"]), reduce=r.choice([None, "f32"]))
        if par == "ddp":
            seg["ddp_bcast"] = r.random() < 0.8
            seg["ddp_gabv"] = r.random() < 0.3

    def gen_save(self, scn, seg, nxt):
        r = self.r
        api = r.choice(["sd", "sd", "stateful", "raw"])
        if api == "raw" and (seg["par"] == "ddp" or (nxt and nxt["par"] == "ddp")):
            api = "sd"
        sv = dict(api=api)
        sv["mode"] = "sharded" if api == "stateful" else r.choice(["sharded", "sharded", "full", "full_offload"])
        sv["ign_frozen"] = bool(scn["model"]["frozen"]) and api != "raw" and r.random() < 0.5
        a = r.random()
        sv["async"] = "none" if a < 0.55 else "thread" if a < 0.95 else "process"
        if sv["async"] != "none":
            sv["mutate"] = r.choice([0, 1, 2])
            # training while a thread-mode save still runs collectives on the same PG deadlocks (see FINDINGS C)
            sv["async_pg"] = "new" if sv["mutate"] else r.choice(["default", "new"])
        sv["writer"] = dict(
            sfpr=r.random() < 0.7,
            threads=r.choice([1, 1, 2, 3]),
            copy_ahead=r.choice([10_000_000, 64]),
            fmt="safetensors" if r.random() < 0.1 else "torch_save",
        )
        sv["planner"] = dict(
            flat=r.random() < 0.9, flat_sh=r.random() < 0.85, dedup_low=r.random() < 0.3, cache=r.random() < 0.2
        )
        sv["presave"] = r.random() < 0.15
        return sv

    def gen_load(self, scn, seg, prev_sv):
        r = self.r
        if prev_sv["api"] == "stateful":
            api = "stateful"
        else:
            api = r.choice(["sd", "sd", "raw"])
            if api == "raw" and (seg["par"] == "ddp" or prev_sv.get("ign_frozen")):
                api = "sd"
        ld = dict(api=api)
        ld["mode"] = "sharded" if api == "stateful" else r.choice(["sharded", "sharded", "full", "full_bcast"])
        if api == "raw" and ld["mode"] == "full_bcast":
            ld["mode"] = "full"
        ld["bcast_offload"] = r.random() < 0.5
        ld["planner"] = dict(flat=prev_sv["planner"]["flat"], flat_sh=r.random() < 0.85)
        return ld

    def gen_seg(self, scn):
        r = self.r
        seg = dict(w=r.randint(1, self.max_w), steps=r.choice([0, 1, 2, 2, 3, 4]))
        self.gen_par(scn, seg)
        seg["resave"] = r.random() < 0.5
        return seg

    def fix_chain(self, scn):
        segs = scn["segs"]
        for i, s in enumerate(segs):
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if nxt is None:
                s.pop("save", None)
            elif "save" not in s or (s["save"]["api"] == "raw" and (s["par"] == "ddp" or nxt["par"] == "ddp")):
                s["save"] = self.gen_save(scn, s, nxt)
            if i == 0:
                s.pop("load", None)
            elif "load" not in s or not self.load_ok(s, segs[i - 1]["save"]):
                s["load"] = self.gen_load(scn, s, segs[i - 1]["save"])

    @staticmethod
    def load_ok(seg, sv):
        ld = seg["load"]
        if (sv["api"] == "stateful") != (ld["api"] == "stateful"):
            return False
        if ld["api"] == "raw" and (seg["par"] == "ddp" or sv.get("ign_frozen")):
            return False
        return ld["planner"]["flat"] == sv["planner"]["flat"]

    def generate(self):
        r = self.r
        scn = dict(seed=r.randint(1, 10**6), dtype="f64" if r.random() < 0.8 else "f32", batch=12)
        scn["model"] = self.gen_model()
        scn["optim"] = self.gen_optim()
        scn["clip"] = r.choice([None, None, None, 0.5, 1.0])
        scn["zg_none"] = r.random() < 0.8
        scn["flat_osd"] = r.random() < 0.2
        scn["segs"] = [self.gen_seg(scn) for _ in range(r.choice([2, 2, 3, 3, 4]))]
        self.fix_chain(scn)
        if not self.valid(scn):
            return self.generate()
        return scn

    def valid(self, scn):
        m = scn["model"]
        if not model_dims_ok(m):
            return False
        for i, s in enumerate(scn["segs"]):
            w = s["w"]
            if s["par"] == "none" and w != 1:
                return False
            if s["par"] in ("fsdp_tp", "hsdp") and s["mesh2d"][0] * s["mesh2d"][1] != w:
                return False
            if s.get("tp"):
                tpd = w if s["par"] == "tp" else s["mesh2d"][1]
                for k, v in s["tp"].items():
                    if k in ("emb", "head"):
                        if m.get("tie") or (k == "emb" and m["kind"] != "tok"):
                            return False
                        continue
                    if int(k) >= len(m["blocks"]):
                        return False
                    b = m["blocks"][int(k)]
                    want = {"lin": ("col", "row"), "mlp": ("mlp",), "attn": ("attn",)}.get(b["t"], ())
                    if v not in want:
                        return False
                    if b["t"] == "attn" and b["nh"] % tpd:
                        return False
            for u in s.get("units", []):
                if u in ("emb", "head"):
                    if m.get("tie") or (u == "emb" and m["kind"] != "tok"):
                        return False
                elif int(u) >= len(m["blocks"]):
                    return False
            if i < len(scn["segs"]) - 1 and "save" not in s:
                return False
            if i > 0 and not self.load_ok(s, scn["segs"][i - 1]["save"]):
                return False
            if s.get("save", {}).get("ign_frozen") and not m["frozen"]:
                return False
        return True

    def mutate(self, parent):
        r = self.r
        for _ in range(20):
            s = copy.deepcopy(parent)
            s.pop("id", None)
            k = r.randrange(9)
            segs = s["segs"]
            i = r.randrange(len(segs))
            if k == 0:
                s["model"] = self.gen_model()
                for sg in segs:
                    self.gen_par(s, sg)
                for sg in segs:
                    sg.pop("save", None)
                    sg.pop("load", None)
            elif k == 1:
                s["optim"] = self.gen_optim()
            elif k == 2:
                self.gen_par(s, segs[i])
            elif k == 3:
                segs[i]["w"] = r.randint(1, self.max_w)
                self.gen_par(s, segs[i])
            elif k == 4 and i < len(segs) - 1:
                segs[i]["save"] = self.gen_save(s, segs[i], segs[i + 1])
                if i + 1 < len(segs):
                    segs[i + 1].pop("load", None)
            elif k == 5 and i > 0:
                segs[i]["load"] = self.gen_load(s, segs[i], segs[i - 1]["save"])
            elif k == 6:
                segs[i]["steps"] = r.choice([0, 1, 2, 3, 4])
                segs[i]["resave"] = not segs[i].get("resave")
            elif k == 7:
                if len(segs) > 2 and r.random() < 0.5:
                    del segs[i]
                    for sg in segs:
                        sg.pop("load", None)
                else:
                    segs.insert(i, self.gen_seg(s))
                    for sg in segs:
                        sg.pop("load", None)
            else:
                s["clip"] = r.choice([None, 0.5, 1.0])
                s["zg_none"] = r.random() < 0.8
                s["flat_osd"] = r.random() < 0.2
                s["seed"] = r.randint(1, 10**6)
            self.fix_chain(s)
            if self.valid(s):
                return s
        return self.generate()

    def simplifications(self, scn):
        segs = scn["segs"]
        out = []

        def c(fn):
            s = copy.deepcopy(scn)
            s.pop("id", None)
            try:
                fn(s)
            except (KeyError, IndexError, ValueError):
                return
            self.fix_chain_min(s)
            if self.valid(s) and json.dumps(s, sort_keys=True) != json.dumps(scn, sort_keys=True):
                out.append(s)

        for i in range(len(segs)):
            if len(segs) > 1:

                def rm(s, i=i):
                    del s["segs"][i]

                c(rm)
        for i in range(len(segs)):
            for n in (0, 1):
                if segs[i]["steps"] > n:
                    c(lambda s, i=i, n=n: s["segs"][i].__setitem__("steps", n))
        for j in range(len(scn["model"]["blocks"])):

            def rb(s, j=j):
                m = s["model"]
                del m["blocks"][j]
                m["frozen"] = [f - (f > j) for f in m["frozen"] if f != j]
                for sg in s["segs"]:
                    if "tp" in sg:
                        sg["tp"] = {
                            (k if k in ("emb", "head") else str(int(k) - (int(k) > j))): v
                            for k, v in sg["tp"].items()
                            if k != str(j)
                        }
                    if "units" in sg:
                        sg["units"] = [
                            (u if u in ("emb", "head") else str(int(u) - (int(u) > j)))
                            for u in sg["units"]
                            if u != str(j)
                        ]

            c(rb)
        m = scn["model"]
        if m.get("tie"):
            c(lambda s: s["model"].pop("tie"))
        if m["frozen"]:

            def uf(s):
                s["model"]["frozen"] = []
                for sg in s["segs"]:
                    if "save" in sg:
                        sg["save"]["ign_frozen"] = False

            c(uf)
        for k, v in (("clip", None), ("zg_none", True), ("flat_osd", False), ("dtype", "f64")):
            if scn.get(k) != v:
                c(lambda s, k=k, v=v: s.__setitem__(k, v))
        o = scn["optim"]
        for k, v in (
            ("name", "sgd"),
            ("foreach", None),
            ("fused", False),
            ("amsgrad", False),
            ("wd", 0.0),
            ("momentum", 0.9),
            ("nesterov", False),
        ):
            if k in o and o[k] != v or (k == "name" and o[k] != v):

                def so(s, k=k, v=v):
                    s["optim"][k] = v
                    if k == "name":
                        s["optim"] = dict(
                            name="sgd",
                            lr=s["optim"]["lr"],
                            wd=0.0,
                            foreach=None,
                            fused=False,
                            momentum=0.9,
                            nesterov=False,
                        )

                c(so)
        for i, sg in enumerate(segs):
            for w in range(1, sg["w"]):

                def sw(s, i=i, w=w):
                    g = s["segs"][i]
                    g["w"] = w
                    if "mesh2d" in g:
                        g["mesh2d"] = [1, w] if g["par"] == "fsdp_tp" else [w, 1]
                    if g["par"] == "none" and w != 1:
                        raise ValueError

                c(sw)
            for p in ("fsdp", "ddp", "none"):
                if sg["par"] != p:

                    def sp(s, i=i, p=p):
                        g = s["segs"][i]
                        for k in ("tp", "units", "raf", "raf_root", "mp", "mesh2d", "ddp_bcast", "ddp_gabv"):
                            g.pop(k, None)
                        g["par"] = p
                        if p == "none":
                            g["w"] = 1

                    c(sp)
            for k in list(sg.get("tp", {})):
                c(lambda s, i=i, k=k: s["segs"][i]["tp"].pop(k))
            for u in list(sg.get("units", [])):
                c(lambda s, i=i, u=u: s["segs"][i]["units"].remove(u))
            for k, v in (
                ("mp", None),
                ("raf", True),
                ("raf_root", True),
                ("ddp_bcast", True),
                ("ddp_gabv", False),
                ("resave", False),
            ):
                if k in sg and sg[k] != v:
                    c(lambda s, i=i, k=k, v=v: s["segs"][i].__setitem__(k, v))
            if "save" in sg:
                sv = sg["save"]
                for k, v in (
                    ("async", "none"),
                    ("api", "sd"),
                    ("mode", "sharded"),
                    ("presave", False),
                    ("ign_frozen", False),
                    ("mutate", 0),
                    ("async_pg", "default"),
                ):
                    if k in sv and sv[k] != v:
                        c(lambda s, i=i, k=k, v=v: s["segs"][i]["save"].__setitem__(k, v))
                for k, v in (("sfpr", True), ("threads", 1), ("copy_ahead", 10_000_000), ("fmt", "torch_save")):
                    if sv["writer"].get(k) != v:
                        c(lambda s, i=i, k=k, v=v: s["segs"][i]["save"]["writer"].__setitem__(k, v))
                for k, v in (("flat", True), ("flat_sh", True), ("dedup_low", False), ("cache", False)):
                    if sv["planner"].get(k) != v:

                        def spl(s, i=i, k=k, v=v):
                            s["segs"][i]["save"]["planner"][k] = v
                            if k == "flat" and i + 1 < len(s["segs"]):
                                s["segs"][i + 1]["load"]["planner"]["flat"] = v

                        c(spl)
            if "load" in sg:
                ld = sg["load"]
                for k, v in (("api", "sd"), ("mode", "sharded"), ("bcast_offload", False)):
                    if ld.get(k) != v:
                        c(lambda s, i=i, k=k, v=v: s["segs"][i]["load"].__setitem__(k, v))
                if ld["planner"].get("flat_sh") is not True:
                    c(lambda s, i=i: s["segs"][i]["load"]["planner"].__setitem__("flat_sh", True))
        return out

    def fix_chain_min(self, scn):
        segs = scn["segs"]
        for i, s in enumerate(segs):
            if i == len(segs) - 1:
                s.pop("save", None)
            elif "save" not in s:
                s["save"] = dict(
                    api="sd",
                    mode="sharded",
                    ign_frozen=False,
                    async_="none",
                    writer=dict(sfpr=True, threads=1, copy_ahead=10_000_000, fmt="torch_save"),
                    planner=dict(flat=True, flat_sh=True, dedup_low=False, cache=False),
                    presave=False,
                )
                s["save"]["async"] = s["save"].pop("async_")
            if s.get("save", {}).get("api") == "raw" and (s["par"] == "ddp" or segs[i + 1]["par"] == "ddp"):
                s["save"]["api"] = "sd"
            if i == 0:
                s.pop("load", None)
            elif "load" not in s:
                s["load"] = dict(
                    api="stateful" if segs[i - 1]["save"]["api"] == "stateful" else "sd",
                    mode="sharded",
                    bcast_offload=False,
                    planner=dict(flat=segs[i - 1]["save"]["planner"]["flat"], flat_sh=True),
                )
            if i > 0:
                sv, ld = segs[i - 1]["save"], s["load"]
                if (sv["api"] == "stateful") != (ld["api"] == "stateful"):
                    ld["api"] = "stateful" if sv["api"] == "stateful" else "sd"
                    ld["mode"] = "sharded" if ld["api"] == "stateful" else ld["mode"]
                if ld["api"] == "raw" and (s["par"] == "ddp" or sv.get("ign_frozen")):
                    ld["api"] = "sd"
                ld["planner"]["flat"] = sv["planner"]["flat"]
