# Contributing

Conventions for this repo, matching the chassis the other portfolio
projects share.

## Stack

- Python 3.13, `uv` for dependency management (`[tool.uv] package = false`,
  this is an application, not a published package).
- `turbine/` is the harness package; `turbine/backends/` holds the backend
  protocol and its mock and OpenAI-compatible implementations.
- `scripts/` holds the status metric and the conventions checker.
- CI runs ruff and pytest on every push to main.

## Working conventions

1. **Plan before code.** For anything beyond a trivial fix, propose the
   approach and get sign-off before writing files.
2. **Tasks come only from BACKLOG.md.** No ad-hoc scope mid-session.
3. **Every nontrivial choice gets a DECISIONS.md entry**, written when the
   choice is made rather than reconstructed later.
4. **Small diffs.** Several focused changes beat one large one.
5. **Tests for every acceptance criterion.** If a task has a defined
   "done", there is a test proving it.
6. **Never touch files outside the current task's scope.**
7. **Update BACKLOG.md checkboxes at the end of each session.**

## Commit granularity

Batch a working session into a single commit, two at most. Commit straight
to the default branch. Never pad the history; a commit should read as one
coherent change with a message that explains it. Imperative mood, no
prefixes, no attribution trailers.

## Writing style

This is a portfolio repo. It should read as though one engineer wrote it,
because one engineer is responsible for it.

**Never use:**

- Em dashes or en dashes. Use a comma, a colon, parentheses, or two
  sentences. The conventions script blocks these.
- Emoji in headings, tables, or status banners.
- Filler adjectives: comprehensive, robust, seamless, powerful,
  cutting-edge, production-grade (unless literally measured), blazing fast.
- "Leverage" as a verb. Use "use".
- The "it's not just X, it's Y" construction.
- "Let's dive in", "in today's landscape", "at its core", "the key insight
  is".
- Bold lead-ins on every bullet in a list. Vary the shape.

**Do:**

- Write plainly and specifically. Name real numbers, real file paths, real
  failures.
- Vary sentence length. Some short.
- Let "What Broke" be genuinely unflattering.
- Prefer concrete verbs over abstractions.

Comments explain why, never what. No narration comments, no section
banners, no docstrings on trivial helpers.

## Secrets

Never commit credentials. This repo is public; keys live in `.env`
(gitignored) and GitHub Actions secrets only. The conventions script scans
for credential shapes as a net, not a substitute.
