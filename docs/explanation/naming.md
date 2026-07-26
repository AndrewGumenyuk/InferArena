# Naming Research

The working project name is **InferArena**. Before committing to it for an open-source release, availability was checked on PyPI and GitHub.

## Availability summary

| Name | PyPI | GitHub org/repo | Notes |
|---|---|---|---|
| relay | Taken (200) | Taken (200) | Existing package and org; would conflict |
| inferarena | Available (404) | Available (404) | Strong candidate, available everywhere checked |
| flowbench | Available (404) | Taken (200) | GitHub repo exists |
| tokenflow | Taken (200) | Taken (200) | Existing projects |
| inferscope | Taken (200) | Taken (200) | Existing profiling tool (MicheleCampi/inferscope) |
| relay-inference | Available (404) | Available (404) | Descriptive fallback |
| inferlab | Available (404) | Taken (200) | GitHub repo exists |
| fluxbench | Available (404) | Taken (200) | GitHub repo exists |

## Final decision

The project and repository are named **InferArena**:

- Repository: `https://github.com/AndrewGumenyuk/InferArena`
- Package name: `inferarena`

## Recommendation

**InferArena** is the cleanest available option:

- Memorable and distinct from existing inference projects.
- Suggests fair comparison, which matches InferArena's experimentation vision.
- Available on both PyPI and GitHub at the time of this check.

If the project stays closer to a systems research platform than a benchmark, **InferArena** still works as a brand for the experimentation harness.

## Caveat

Availability was checked with HTTP status codes on a single date. Domain availability was not verified. Re-check before creating public repositories or publishing packages.
