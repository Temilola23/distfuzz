# Writing and reviewing pull requests

## Writing a PR

**One concern per PR.** A new mutator and a fix to the reference model are two PRs. Moving code and changing it are two commits at least, so the reviewer can check the move with `git diff -M` and read the change on its own.

**Size.** Aim for a diff someone can review in 20 minutes, roughly 400 changed lines excluding moved files and test data. If it has to be larger, say in the description which files to read first.

**Title.** A conventional commit, because it becomes the squash commit on `main`: `fix(dtensor): normalise variable names in signatures`.

**Description.** Use the template. The parts reviewers read most:

- *Summary*: what changes for someone running the fuzzer.
- *Test Plan*: the commands you ran and what they printed. "111 passed, 138 deselected" is useful; "tests pass" is not. For multi-rank changes, say which torch version the container had.
- *Findings impact*: whether the change adds, removes or renames findings. A reference-model fix that removes false positives should say how many and on which run.
- *Notes*: shortcuts you took and what you left for later.

**Before requesting review**

- `make check` passes on the host.
- `make docker-test` passes if you touched anything that runs ranks.
- CI is green. A red CI is not ready for review.
- You read your own diff once on GitHub.

## Reviewing a PR

Read the description, then the tests, then the code. Check, in this order:

1. **Correctness of the oracle.** Could this change report a PyTorch bug that is really a harness bug, or hide a real one? Look for comparisons that skip values, tolerances that grew, and exceptions that are swallowed.
2. **Rank symmetry.** Does every rank reach the same collectives? A rank that returns early or raises alone makes the others hang until timeout.
3. **Determinism.** Same seed, same programs. Watch for `random` calls without the passed-in `rng`, dict ordering that depends on hashing, and time-based decisions.
4. **Cleanup.** Process groups destroyed, rank processes killed on timeout, temporary directories removed. Leaks show up as slowdowns an hour into a campaign.
5. **Tests.** Does a test fail without the change? For a fix, ask for the failing case first.
6. **Scope.** Unrelated refactors belong in another PR.

Write comments as questions or concrete suggestions, and mark the ones that block merging. Approve when the blocking comments are resolved; nits can be follow-ups.

## Merging

- Squash-merge once CI is green and the blocking comments are resolved.
- The squash commit message is the PR title plus a short body; delete the branch after merging (the repo does this automatically).
- If `main` moved, rebase and let CI run again before merging.
