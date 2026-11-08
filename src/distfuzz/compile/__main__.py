import sys


def main():
    cmds = ("fuzz", "minimize", "repro", "replay")
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"usage: python -m distfuzz.compile {{{','.join(cmds)}}} ...", file=sys.stderr)
        sys.exit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "fuzz":
        from distfuzz.compile.fuzz import main as run
    elif cmd == "minimize":
        from distfuzz.compile.minimize import main as run
    elif cmd == "replay":
        from distfuzz.compile.fuzz import replay_main as run
    else:
        from distfuzz.compile.repro import main as run
    run(rest)


main()
