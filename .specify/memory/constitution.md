<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Modified principles: placeholders -> Collaboration First; Spec Before Code; Demo Before Scale; Reuse Before Reinvention; Reproducible Research; Written Learning Loop
- Added sections: Project Constraints; Development Workflow & Quality Gates
- Removed sections: none
- Templates requiring updates: ✅ .specify/templates/plan-template.md (compatible), ✅ .specify/templates/spec-template.md (compatible), ✅ .specify/templates/tasks-template.md (compatible)
- Follow-up TODOs: none
-->
# baseline-safe-marl Constitution

## Core Principles

### I. Collaboration First
All non-trivial work MUST flow through explicit team coordination. Godclaw owns task decomposition,
review, integration, and reporting. phd1 and phd2 MUST work through assigned tasks, keep scope tight,
and return structured progress or blockers instead of making silent strategic changes.

### II. Spec Before Code
Any non-trivial feature, experiment pipeline, algorithm integration, benchmark adapter, or paper section
MUST begin from a written spec or plan artifact before implementation. Code MAY start only after the
relevant scope, success criteria, and boundaries are written clearly enough for review.

### III. Demo Before Scale
The team MUST begin from the smallest runnable demo before attempting large-scale training,
full-benchmark coverage, or multi-algorithm optimization. The required order is: environment smoke test
-> minimal rollout -> single-algorithm prototype -> comparable baseline runs -> broader optimization.

### IV. Reuse Before Reinvention
The team MUST search existing GitHub repositories, papers, and reference implementations before building
core components from scratch. Reuse is preferred when it reduces risk, accelerates validation, and keeps
behavior inspectable. Any decision to re-implement a known component SHOULD be justified in writing.

### V. Reproducible Research
Every experiment-related change MUST aim for reproducibility and comparison. Configurations,
dependencies, benchmark assumptions, algorithm versions, and evaluation commands MUST be recorded well
enough for another team member to rerun the result. Claims without a reproducible path do not count as
completed research work.

### VI. Written Learning Loop
The project MUST maintain two written support logs outside the code path: a blocker log for unresolved
external/environment issues and an error notebook for lessons learned, mistakes, fixes, and process
improvements. Repeated mistakes are treated as process failures until documented and addressed.

## Project Constraints

- Project scope: multi-agent Safe RL baseline with six as-new-as-reasonable algorithms on multi-agent
  safe mujoco.
- Primary priority: team Git/GitHub collaboration workflow first, research quality close behind.
- Project layout MUST remain organized: code, specs, architecture notes, experiment assets, and paper
  materials are kept separate and readable.
- Python environments MAY be managed with uv inside the project.
- Any system-level change the team cannot safely complete alone (sudo, system packages, service config,
  privileged drivers, cluster policy changes) MUST be documented and escalated instead of improvised.
- Paper writing is part of the project deliverable; LaTeX sources live under `docs/paper/`.

## Development Workflow & Quality Gates

1. Start from a written spec and success criteria.
2. Split work into small tasks that can be reviewed independently.
3. Prefer one minimal runnable prototype before broader engineering.
4. Require review before merging meaningful feature work to the main branch.
5. Record blockers in `~/Documents/godclaw/obsidian/drafts/baseline-safe-marl-环境与阻塞问题.md` when
   the team cannot self-resolve them.
6. Record mistakes and process improvements in
   `~/Documents/godclaw/obsidian/drafts/baseline-safe-marl-错题本.md`.
7. Keep paper, specs, and code evolution aligned enough that research outputs stay auditable.

## Governance

This constitution supersedes ad-hoc project habits for baseline-safe-marl. Amendments require a written
change in this file, an explanation of why the old rule is insufficient, and review by Godclaw. Any plan,
spec, or task list that conflicts with this constitution is invalid until updated. Compliance checks MUST
happen during planning, review, and merge decisions.

**Version**: 1.0.0 | **Ratified**: 2026-03-30 | **Last Amended**: 2026-03-30
