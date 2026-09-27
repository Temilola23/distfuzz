# python tcpstore_race.py [seconds=60] [clients=8]  (Docker only; TCPStore stress test, has found no crash so far)
from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import time
from datetime import timedelta

MEMCORRUPT = {-11: "SIGSEGV", -6: "SIGABRT", -4: "SIGILL", -8: "SIGFPE", -7: "SIGBUS"}


def _client(host, port, cid, seconds, seed):
    import torch.distributed as dist

    rng = random.Random(seed)
    store = dist.TCPStore(host, port, world_size=None, is_master=False, timeout=timedelta(seconds=2))
    keys = [f"k{i}" for i in range(8)]
    end = time.time() + seconds
    ops = 0
    while time.time() < end:
        k = rng.choice(keys)
        op = rng.random()
        try:
            if op < 0.30:
                store.set(k, str(rng.randrange(1 << 30)))
            elif op < 0.45:
                store.get(k)
            elif op < 0.60:
                store.add(k, rng.randrange(-3, 4))
            elif op < 0.72:
                store.compare_set(k, str(rng.randrange(4)), str(rng.randrange(4)))
            elif op < 0.80:
                store.delete_key(k)
            elif op < 0.88:
                store.num_keys()
            elif op < 0.95:
                # wait with a short timeout on keys that may never be set -> expect timeout, not hang/segv
                store.wait([rng.choice(keys)], timedelta(milliseconds=rng.randrange(1, 60)))
            else:
                store.check([rng.choice(keys)])
        except Exception:  # noqa: BLE001 - Python errors / timeouts are fine
            pass
        ops += 1
    return ops


def _server(host, port, seconds):
    import torch.distributed as dist

    srv = dist.TCPStore(host, port, world_size=None, is_master=True, timeout=timedelta(seconds=5))
    # keep the server object alive and also act as an extra racing client
    _client(host, port, -1, seconds, 999)
    del srv


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    nclients = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    import torch.distributed as dist  # noqa: F401

    host, base_port = "127.0.0.1", 29700 + random.randrange(200)
    ctx = mp.get_context("spawn")

    findings = []
    round_seconds = min(seconds, 30)
    rounds = max(1, int(seconds // round_seconds))
    for rnd in range(rounds):
        port = base_port + rnd  # fresh port each round to avoid TIME_WAIT reuse
        srv = ctx.Process(target=_server, args=(host, port, round_seconds), daemon=True)
        srv.start()
        time.sleep(0.5)
        procs = [
            ctx.Process(target=_client, args=(host, port, c, round_seconds, rnd * 100 + c), daemon=True)
            for c in range(nclients)
        ]
        for p in procs:
            p.start()
        deadline = time.time() + round_seconds + 30
        for p in procs + [srv]:
            p.join(timeout=max(1, deadline - time.time()))
        for i, p in enumerate(procs + [srv]):
            if p.is_alive():
                findings.append(f"HANG: proc {i} still alive past deadline (round {rnd})")
                p.kill()
            elif p.exitcode in MEMCORRUPT:
                findings.append(f"CRASH: proc {i} exit {p.exitcode} ({MEMCORRUPT[p.exitcode]}) round {rnd}")
    print("TCPSTORE_ROUNDS", rounds, "CLIENTS", nclients)
    if findings:
        for f in findings:
            print("FINDING", f)
    else:
        print("TCPSTORE_NO_FINDINGS (clean errors/timeouts only)")


if __name__ == "__main__":
    main()
