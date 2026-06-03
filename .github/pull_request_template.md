## Summary

<!-- One-paragraph description of WHAT changed and WHY. -->

## Motivation

<!-- What problem does this solve? Link related issues, Jira tickets, or upstream changes. -->

## Changes

<!-- Bullet list of the concrete changes. Group by area when touching multiple components. -->

### Test framework / runner
- 

### Test configs / scenarios
- 

### Dependencies
- 

### Documentation
- 

## Breaking changes

<!-- List any backward-incompatible changes: renamed env vars, removed CLI flags, changed JSON schema keys, etc. Use "None" if not applicable. -->

- 

## How to test

<!-- Step-by-step instructions a reviewer can follow to verify correctness. -->

```bash
# example
./run_tests.sh --dry-run "autorag and positive"
```

## Checklist

- [ ] `run_tests.sh --dry-run` produces correct commands for affected suites
- [ ] Env example files (`.env.*.example`) updated if env vars changed
- [ ] README / sub-README documentation updated
- [ ] No secrets or credentials committed
- [ ] `uv.lock` regenerated if `pyproject.toml` changed
