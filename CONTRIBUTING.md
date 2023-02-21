# Contributing to VAST

First, thanks for your interest in contributing to VAST! We welcome and
appreciate all contributions, including bug reports, feature suggestions,
tutorials/blog posts, and code improvements.

If you're unsure where to start, we recommend our [`good first issue`](https://github.com/trailofbits/vast/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) issue label.

## Bug reports and feature suggestions

Bug reports and feature suggestions can be submitted to our [issue tracker](https://github.com/trailofbits/vast/issues).

When reporting a bug please provide **a minimal** example with steps to reproduce the issue
if possible. It helps us a lot, as we can get to the bottom of the issue much faster and can
even use it as a test case to catch future regressions.

## Questions

Questions can be submitted to the [discussion page](https://github.com/trailofbits/vast/discussions).

## Legal
For legal reasons, we require contributors to sign our [Contributor License
Agreement](https://cla-assistant.io/trailofbits/vast).  This will be
automatically checked as part of our CI.

## Git & Pull Requests

VAST uses the pull request contribution model. Please make an account on
Github, fork this repo, and submit code contributions via pull request. For
more documentation, look [here](https://guides.github.com/activities/forking/).

Since VAST does not squash commits in a pull request, it is important to uphold
some culture when it comes to commits.

- Commit should ideally be one simple change.
- Commit messages follow a simple format:
  `component: Simple sentence with a dot.` with maximum of 80 chars and optional longer
  message.
- When unsure what component commit modifies, run `git log` on the modified file(s).
- Commits should modify only one component (as a result the project does not have
  to build with each separate commit)
- If you are having troubles coming up with a simple sentence as a commit message,
  that is short enough, it may be a good indicator that the commit should be split.

Some pull request guidelines:

- Minimize irrelevant changes (formatting, whitespace, etc) to code that would
  otherwise not be touched by this patch. Save formatting or style corrections
  for a separate pull request that does not make any semantic changes.
- When possible, large changes should be split up into smaller focused pull
  requests.
- Fill out the pull request description with a summary of what your patch does,
  key changes that have been made, and any further points of discussion, if
  applicable.
- Title your pull request with a brief description of what it's changing.
  "Fixes #123" is a good comment to add to the description, but makes for an
  unclear title on its own.
- CI must pass for the PR to be merged.
- There must be a review from some maintainer that accepts changes for the PR to be merged.
