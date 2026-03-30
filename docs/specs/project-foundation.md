# Project Specification: baseline-safe-marl Foundation

**Created**: 2026-03-30  
**Status**: Draft

## Objective

Build a collaboration-first, spec-driven baseline repository for **multi-agent Safe RL** research on
**multi-agent safe mujoco**, with six as-new-as-reasonable algorithm candidates, while ensuring the team
can progress through small runnable milestones instead of jumping directly into large training jobs.

## User Scenarios & Testing

### User Story 1 - Team can start from a minimal runnable benchmark demo (Priority: P1)

As the project lead, I need the repository to support a tiny benchmark smoke test so the team can verify
that the environment, dependency chain, and basic rollout path work before touching expensive training.

**Why this priority**: Without a runnable environment demo, all later algorithm work is high-risk guesswork.

**Independent Test**: A team member can set up the environment and run a script that performs reset,
step, and short rollout on the benchmark without training a full algorithm.

**Acceptance Scenarios**:
1. **Given** a clean project checkout, **When** a teammate follows the benchmark quickstart,
   **Then** they can run a smoke test that resets the environment and completes several valid steps.
2. **Given** the smoke test script, **When** the environment dependencies are available,
   **Then** the script exits with explicit success or a structured failure reason.

---

### User Story 2 - Team can choose an algorithm shortlist with evidence (Priority: P1)

As the project lead, I need a documented shortlist of six algorithm candidates so the team can avoid
random implementation drift and align on what will actually count as the baseline.

**Why this priority**: Algorithm choice drives all later architecture, integration, and experiment work.

**Independent Test**: The shortlist document can be reviewed on its own and clearly explains inclusion,
exclusion, reuse potential, and benchmark fit.

**Acceptance Scenarios**:
1. **Given** the shortlist document, **When** a reviewer checks it,
   **Then** each candidate includes recency, implementation availability, and benchmark fit.
2. **Given** multiple candidate algorithms, **When** they are compared,
   **Then** the team can identify which algorithm should be prototyped first.

---

### User Story 3 - Team can develop under a clean SDD + Git workflow (Priority: P1)

As the project lead, I need every substantial task to pass through spec, small-task execution, review,
and merge so that the project trains team collaboration rather than only code generation.

**Why this priority**: Collaboration discipline is an explicit project goal, not a side effect.

**Independent Test**: A new feature or experiment task can be traced from spec to branch to reviewable
changes without ambiguity.

**Acceptance Scenarios**:
1. **Given** a non-trivial task, **When** development begins,
   **Then** there is a written spec or plan before code changes expand.
2. **Given** work from phd1 or phd2, **When** it is ready,
   **Then** it reaches Godclaw in a reviewable form instead of being merged silently.

---

### User Story 4 - Project can grow toward research-grade outputs (Priority: P2)

As the project lead, I need the repository to support docs, experiments, and LaTeX paper materials so
that research outputs stay organized while the codebase evolves.

**Why this priority**: This is a baseline project, not just a throwaway script collection.

**Independent Test**: The repository contains clear homes for specs, architecture notes, experiment assets,
and paper files.

**Acceptance Scenarios**:
1. **Given** the repository structure, **When** a teammate looks for paper sources,
   **Then** they can find them under `docs/paper/`.
2. **Given** an experiment or algorithm note, **When** it is documented,
   **Then** it lands in a stable project location rather than scattered ad hoc.

## Edge Cases

- What happens when benchmark installation succeeds on one machine but fails on another due to system
  dependencies?
- How does the team proceed if a promising algorithm has no usable public implementation?
- What happens when a benchmark wrapper runs but training is too slow or unstable for quick feedback?
- How do we prevent heavyweight tasks from blocking all team progress?

## Requirements

### Functional Requirements

- **FR-001**: The repository MUST support a minimal benchmark smoke-test path before full training.
- **FR-002**: The project MUST maintain spec-driven documentation for meaningful development tasks.
- **FR-003**: The team MUST maintain a documented shortlist of six algorithm candidates for the baseline.
- **FR-004**: The project MUST separate code, specs, architecture notes, experiment assets, and paper
  materials into clear directories.
- **FR-005**: The workflow MUST support small independent tasks that can be assigned, reviewed, and merged.
- **FR-006**: The team MUST document unresolved environment/system blockers in the dedicated drafts blocker log.
- **FR-007**: The team MUST document major mistakes, fixes, and process improvements in the error notebook.
- **FR-008**: The first development phase MUST prioritize a runnable prototype over large-scale training.
- **FR-009**: The project MUST preserve enough reproducibility information for benchmark setup and early
  experiment replay.

### Key Entities

- **Benchmark Adapter**: The project-facing entry point that sets up and interacts with multi-agent safe mujoco.
- **Algorithm Candidate**: A shortlisted Safe RL method considered for inclusion in the baseline.
- **Prototype Run**: The smallest reproducible runnable path that proves environment and integration viability.
- **Blocker Record**: A structured unresolved problem that requires escalation or later coordination.
- **Learning Record**: A documented mistake/fix/improvement entry in the project error notebook.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The team can produce one runnable benchmark smoke test before starting heavy training work.
- **SC-002**: The team can document and defend an initial shortlist of six algorithm candidates with source evidence.
- **SC-003**: Every non-trivial implementation task in the first phase can be traced to a written spec or plan.
- **SC-004**: The repository maintains a stable, reviewable structure for code, docs, and paper materials.
- **SC-005**: At least one first-stage prototype path can be handed from planning to review without blocking on
  unresolved heavyweight infrastructure.

## Assumptions

- The GitHub remote repository will be used as the collaboration hub after the local repository is initialized.
- The team may reuse credible public implementations where licenses and code quality allow it.
- Early progress should favor smoke tests, adapters, and thin prototypes rather than full benchmark sweeps.
- Paper drafting will begin before all experiments are complete, but it will stay aligned with the evolving baseline.
