import sys

from distfuzz.dcp import fuzz, minimize, report, standalone

USAGE = """usage: python -m distfuzz.dcp COMMAND [ARGS]

  fuzz [--mode guided|random] [--minutes M] [--out DIR] [--seed S]   run a campaign
  run SCENARIO.json                        replay one scenario in fresh worlds (exit 1 on a finding)
  minimize FINDINGS.jsonl OUT_DIR [SIG..]  shrink findings and emit standalone repros
  report RUN_DIR                           sort a run's signatures into triage buckets"""

COMMANDS = {"fuzz": fuzz.main, "run": standalone.main, "minimize": minimize.main, "report": report.main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(USAGE)
    COMMANDS[sys.argv[1]](sys.argv[2:])
