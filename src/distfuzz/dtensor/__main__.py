import sys

from distfuzz.dtensor import fuzz, minimize, report

COMMANDS = {"fuzz": fuzz.main, "minimize": minimize.main, "report": report.main}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m distfuzz.dtensor {{{'|'.join(COMMANDS)}}} [-h] ...")
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"python -m distfuzz.dtensor {cmd}"
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
