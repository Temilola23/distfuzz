from __future__ import annotations

import zlib


def salt(idx: int, role: str, k: int = 0) -> int:
    return zlib.crc32(f"{idx}:{role}:{k}".encode()) & 0xFFFFFF


def resolve_root(args, members, rank):
    root = args["root"]
    if args.get("root_mode") == "group":
        return members[root] if members and 0 <= root < len(members) else None
    return root if members and root in members else None


def a2a_shapes(args, n, grank):
    spec = args["inspec"]
    rest = list(spec["shape"][1:])
    k = spec["shape"][0] if spec["shape"] else 1
    if args.get("even"):
        return [n * k] + rest, [n * k] + rest, None, None
    M = args["matrix"]
    ins = list(M[grank]) if grank < len(M) else []
    outs = [row[grank] if grank < len(row) else 0 for row in M]
    return [sum(ins)] + rest, [sum(outs)] + rest, ins, outs


def list_len(args, n):
    return max(0, n + int(args.get("n_delta", 0)))
