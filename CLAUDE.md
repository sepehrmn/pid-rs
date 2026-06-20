# CLAUDE.md

Project guidance for Claude Code in **pid-rs**.

## Commit & attribution policy (IMPORTANT)

- **Never add yourself (Claude) or any AI/agent as a commit or PR co-author.** Do not append a
  `Co-Authored-By: Claude …` (or any AI/agent) trailer, and do not add "Generated with Claude Code"
  or similar to commit messages or PR descriptions. Commits are authored **solely by the human
  contributor**.
- **Do not sign commits or tags.** The repo sets `commit.gpgsign=false` / `tag.gpgsign=false`
  locally; keep commits unsigned.
- This is enforced by `attribution.commit` / `attribution.pr` being empty in
  [`.claude/settings.json`](.claude/settings.json). Do not change that.

## Project guide

The full agent/contributor guide — workspace layout, the exact build/test/lint commands (mirroring
CI), `pid-python`/maturin notes, and the numerical conventions to preserve — lives in
[AGENTS.md](AGENTS.md):

@AGENTS.md
