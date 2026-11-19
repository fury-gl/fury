# AGENTS Instructions

This file contains guidance for AI agents and other AI-assisted editors working
on FURY. It complements the
[Automated Contributions Policy](CONTRIBUTING.rst#automated-contributions-policy),
which applies to every contribution.

## REQUIRED: AI usage disclosure

Every pull request description MUST fill the "AI Usage Disclosure" section of
the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): list what the
AI tool was used for and name the tool.

Until a human has reviewed the changes, also add this note at the end of every
pull request description or work summary you generate:

> This pull request includes code written with the assistance of AI.
> The code has **not yet been reviewed** by a human.

This is a mandatory requirement, not optional. Pull requests without the
disclosure are closed without review.

The "Introduce Yourself" section is also required unless the author is a FURY
maintainer or collaborator. Do not write it on behalf of the author: ask them
how they use FURY and what prompted the contribution, and use their words.

## Working on an issue

Before working on any issue, run `gh issue view <number> --repo fury-gl/fury`
to check its labels. Do not open a pull request against an issue labeled
`state: needs triage`. Such pull requests are closed without review until a
maintainer removes the label.

Never open issues or pull requests in a fully-automated way. A human must
review, understand and test every change, and be able to explain it on request.

## Project conventions

- Follow the
  [FURY coding style](CONTRIBUTING.rst#coding-style-and-pre-commit-checks):
  PEP8, numpy docstring style, and keyword-only arguments for every parameter
  with a default value (PEP 3102).
- Prefix commit messages with one of the
  [FURY commit codes](CONTRIBUTING.rst#commit-message-codes) (`BF:`, `NF:`,
  `RF:`, `OPT:`, `CI:`, `MNT:`, `DOC:`, `TEST:`, `STYLE:`, ...).
- Add tests next to the module you change (e.g. `fury/tests/`) and run them
  with `pytest fury` before submitting.
- Run `pre-commit run --all-files` before submitting.
- Fill every section of the pull request template.

## Generated summaries

When generating a summary of your work:

- Describe the "why" of the changes, and why the proposed solution is the
  right one.
- Highlight the parts of the change that require careful review.
- Keep it short. More text is not better. Avoid flattery, avoid stating the
  obvious, avoid filler phrases, and prefer technical clarity over marketing
  tone.
