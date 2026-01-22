# Parhelia Execution Strategy

**Date**: 2026-01-21
**Scope**: SPEC-20 (UX Excellence) + SPEC-21 (Control Plane State)

## Executive Summary

Rather than executing two epics sequentially, we merge them into a unified 6-wave execution plan that:
1. **Prioritizes foundation** - Control plane state prevents future cost incidents
2. **Maximizes parallelization** - Independent work streams run concurrently
3. **Eliminates duplication** - Combined CLI and MCP work where specs overlap
4. **Delivers incremental value** - Each wave is usable and testable

## Analysis

### Overlap Detection

| SPEC-21 Phase | SPEC-20 Phase | Overlap |
|---------------|---------------|---------|
| P5: CLI Introspection | P1: Command Foundation | Both restructure CLI |
| P6: MCP Integration | P3: Agent Excellence | Both expand MCP tools |
| Events table | P4: Real-Time Events | Both need event persistence |

### Dependency Graph

```
SPEC-21 P1 (Data Model) ─────────────────────────────────────┐
         │                                                    │
         ▼                                                    │
SPEC-21 P2 (Registration) ───┐                               │
         │                    │                               │
         ▼                    ▼                               │
SPEC-21 P3 (Heartbeats) ─────┴──▶ SPEC-21 P4 (Reconciler) ──┤
                                           │                  │
         ┌─────────────────────────────────┘                  │
         ▼                                                    ▼
SPEC-21 P5+P6 (CLI/MCP) ◀──────────────────────────── SPEC-20 P3 (Agent)
         │                                                    │
         ▼                                                    ▼
SPEC-20 P4 (Real-Time) ◀─────────────────────────────────────┘
         │
         ▼
SPEC-20 P5 (Interactive)


SPEC-20 P1 (Commands) ─── Independent, can run parallel with SPEC-21 P1
SPEC-20 P2 (Feedback) ─── Independent, can run parallel with SPEC-21 P4
```

### Critical Path

```
Data Model → Registration → Heartbeats → Reconciler → CLI/MCP → Real-Time → Polish
   (P1)         (P2)          (P3)         (P4)       (P5+P6)     (P4')      (P5')
```

Phases not on critical path can be parallelized.

---

## Unified Execution Plan

### Wave 1: Foundation (Parallel Tracks)

**Duration estimate**: 1 sprint

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   Track A: Data Model       │     │   Track B: CLI Structure    │
│   (SPEC-21 P1)              │     │   (SPEC-20 P1)              │
├─────────────────────────────┤     ├─────────────────────────────┤
│ • containers table + schema │     │ • Restructure commands      │
│ • events table + schema     │     │ • Implement aliases         │
│ • heartbeats table + schema │     │ • Shell completion          │
│ • Extend workers table      │     │ • Fuzzy matching            │
│ • StateStore class          │     │ • Expand error codes        │
│ • Database migrations       │     │                             │
└─────────────────────────────┘     └─────────────────────────────┘
```

**Why parallel**: No dependencies between data model and command naming.

**Deliverables**:
- Database schema ready for container tracking
- Consistent `parhelia <noun> <verb>` command structure
- Tab completion working

**Exit criteria**:
- `parhelia db migrate` creates all tables
- `parhelia task list`, `parhelia session attach` work
- Shell completion installs successfully

---

### Wave 2: Container Lifecycle

**Duration estimate**: 1 sprint

```
┌─────────────────────────────────────────────────────────────────┐
│   Track A: Registration & Heartbeats (SPEC-21 P2+P3)            │
├─────────────────────────────────────────────────────────────────┤
│ • Modify dispatch to register containers with Modal sandbox IDs │
│ • Store container_id in workers table                           │
│ • Link containers to tasks and sessions                         │
│ • Persist heartbeats to database                                │
│ • Update container health on heartbeat receipt                  │
│ • Emit events on state changes                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why sequential**: Depends on Wave 1 Track A (data model).

**Deliverables**:
- Every dispatched task creates a container record
- Modal sandbox IDs persisted and queryable
- Heartbeat history available in database

**Exit criteria**:
- `SELECT * FROM containers WHERE task_id = ?` returns sandbox ID
- `SELECT * FROM heartbeats WHERE container_id = ?` returns history
- Container health status updates on heartbeat

---

### Wave 3: Reconciliation & Feedback (Parallel Tracks)

**Duration estimate**: 1 sprint

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   Track A: Reconciler       │     │   Track B: Feedback         │
│   (SPEC-21 P4)              │     │   (SPEC-20 P2)              │
├─────────────────────────────┤     ├─────────────────────────────┤
│ • Modal API client          │     │ • Spinner/progress system   │
│ • ContainerReconciler class │     │ • Progress callbacks        │
│ • Orphan detection          │     │ • Success response format   │
│ • State drift correction    │     │ • Proactive validation      │
│ • Stale container detection │     │ • Enhanced error recovery   │
│ • Background process        │     │                             │
└─────────────────────────────┘     └─────────────────────────────┘
```

**Why parallel**: Reconciler and feedback system are independent.

**Deliverables**:
- Background reconciler syncing with Modal every 60s
- Orphan containers detected and flagged
- All long operations show progress indicators
- Errors include 2+ recovery suggestions

**Exit criteria**:
- Manually create orphan in Modal → reconciler detects within 2 min
- Run `parhelia task create` → spinner visible during dispatch
- Trigger E200 error → see 2+ recovery commands

---

### Wave 4: Introspection (Combined)

**Duration estimate**: 1.5 sprints

```
┌─────────────────────────────────────────────────────────────────┐
│   Combined: CLI + MCP Introspection (SPEC-21 P5+P6 + SPEC-20 P3)│
├─────────────────────────────────────────────────────────────────┤
│ CLI Commands:                                                   │
│ • parhelia containers [list|show|events|health|watch|terminate] │
│ • parhelia reconciler status                                    │
│                                                                 │
│ MCP Tools (expand to 15+):                                      │
│ • parhelia_containers (list, show, terminate)                   │
│ • parhelia_health (control plane status)                        │
│ • parhelia_events (query event history)                         │
│ • parhelia_task_* (create, list, show, cancel, retry)          │
│ • parhelia_session_* (list, attach_info, kill)                 │
│ • parhelia_checkpoint_* (create, list, restore, diff)          │
│ • parhelia_budget_* (status, estimate)                         │
│                                                                 │
│ Efficiency:                                                     │
│ • Add `fields` parameter for selective retrieval               │
│ • Integrate cost estimation in task operations                  │
│ • Budget warnings in responses                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why combined**: Both specs add CLI commands and MCP tools. Building together avoids duplicate patterns.

**Deliverables**:
- Full container visibility via CLI and MCP
- 15+ MCP operations (up from 5)
- Token-efficient responses with field selection
- Cost visible before task execution

**Exit criteria**:
- `parhelia containers` shows all running containers with Modal IDs
- MCP `parhelia_containers` returns same data
- `parhelia_task_create` includes `estimated_cost_usd`
- Field selection reduces response size by >50%

---

### Wave 5: Real-Time & Streaming

**Duration estimate**: 1.5 sprints

```
┌─────────────────────────────────────────────────────────────────┐
│   Combined: Events Architecture (SPEC-20 P4 + SPEC-21 events)   │
├─────────────────────────────────────────────────────────────────┤
│ Event Persistence (builds on Wave 1 events table):              │
│ • EventLogger writes to database                                │
│ • Event replay capability                                       │
│ • Event export (JSONL)                                          │
│                                                                 │
│ Streaming:                                                      │
│ • Streaming MCP notifications (JSON-RPC)                        │
│ • Modal webhook integration (or adaptive polling fallback)      │
│ • Event aggregation (5% intervals, 5s minimum)                  │
│                                                                 │
│ Filtering:                                                      │
│ • --events flag for type filtering                              │
│ • --level flag for severity filtering                           │
│ • --quiet mode (completion only)                                │
│ • --from-start for replay                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Why combined**: Event persistence from SPEC-21 enables event features from SPEC-20.

**Deliverables**:
- Events persisted to database and queryable
- MCP clients receive streaming notifications
- Event latency <500ms (push or adaptive polling)
- Watch commands support filtering and replay

**Exit criteria**:
- `parhelia events replay task-123` shows historical events
- MCP streaming delivers status change within 500ms
- `parhelia task watch --events status,completion` filters correctly

---

### Wave 6: Interactive Polish

**Duration estimate**: 0.5 sprint

```
┌─────────────────────────────────────────────────────────────────┐
│   SPEC-20 P5: Interactive Intelligence                          │
├─────────────────────────────────────────────────────────────────┤
│ • Smart prompts with intelligent defaults                       │
│ • Contextual help system (parhelia help <topic>)               │
│ • Error-specific help (parhelia help E200)                     │
│ • Example-based documentation (parhelia examples <topic>)       │
│ • Preference caching (~/.cache/parhelia/)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Why last**: Polish layer that benefits from stable underlying system.

**Deliverables**:
- Interactive mode with smart defaults
- Help for every error code
- Example workflows documented

**Exit criteria**:
- `parhelia task create` prompts with last-used values as defaults
- `parhelia help E200` shows SESSION_NOT_FOUND details
- `parhelia examples gpu` shows GPU task examples

---

## Issue Structure

Reorganize beads issues to match waves:

```
ph-zko (Epic: Control Plane) ─┬─ ph-w1a: Wave 1 Track A - Data Model
                              ├─ ph-w2a: Wave 2 - Container Lifecycle
                              ├─ ph-w3a: Wave 3 Track A - Reconciler
                              └─ (merge into combined waves)

ph-fj8 (Epic: UX Excellence) ─┬─ ph-w1b: Wave 1 Track B - CLI Structure (was ph-eie)
                              ├─ ph-w3b: Wave 3 Track B - Feedback (was ph-fla)
                              └─ ph-w6: Wave 6 - Interactive Polish (was ph-o8q)

New Combined Issues:
  ph-w4: Wave 4 - Introspection (combines SPEC-21 P5+P6 + SPEC-20 P3)
  ph-w5: Wave 5 - Real-Time & Streaming (combines SPEC-20 P4 + events)
```

### Dependency Graph (Issues)

```
Wave 1A (Data Model) ──┬──▶ Wave 2 (Container Lifecycle) ──▶ Wave 3A (Reconciler) ──┐
                       │                                                             │
                       └──────────────────────────────────────────────────────────────┼──▶ Wave 4 (Introspection)
                                                                                     │           │
Wave 1B (CLI) ─────────────────────────────────────────────────────────────────────┘           │
                                                                                                │
Wave 3B (Feedback) ─────────────────────────────────────────────────────────────────────────────┤
                                                                                                │
                                                                                                ▼
                                                                                         Wave 5 (Events)
                                                                                                │
                                                                                                ▼
                                                                                         Wave 6 (Polish)
```

---

## Execution Order

| Order | Wave | Tracks | Dependencies | Can Start After |
|-------|------|--------|--------------|-----------------|
| 1 | Wave 1A | Data Model | None | Immediately |
| 1 | Wave 1B | CLI Structure | None | Immediately |
| 2 | Wave 2 | Container Lifecycle | Wave 1A | Wave 1A complete |
| 3 | Wave 3A | Reconciler | Wave 2 | Wave 2 complete |
| 3 | Wave 3B | Feedback | None | Immediately (or after Wave 1B) |
| 4 | Wave 4 | Introspection | Wave 3A, Wave 1B | Wave 3A complete |
| 5 | Wave 5 | Real-Time | Wave 4 | Wave 4 complete |
| 6 | Wave 6 | Polish | Wave 5 | Wave 5 complete |

**Parallelization opportunities**:
- Wave 1A ∥ Wave 1B
- Wave 3A ∥ Wave 3B
- Wave 3B can start with Wave 1 (no hard dependency)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Modal API changes | High | Abstract behind ModalClient interface |
| Schema migrations on existing data | Medium | Test migrations on copy of prod DB |
| Reconciler performance at scale | Medium | Batch operations, configurable intervals |
| MCP streaming complexity | Medium | Fallback to polling if streaming fails |
| CLI breaking changes | Medium | Maintain aliases for old command names |

---

## Success Metrics

### After Wave 2
- 100% of containers have Modal sandbox IDs in database
- Zero orphan containers possible without detection

### After Wave 4
- `parhelia containers` shows accurate state within 2 minutes of reality
- 15+ MCP operations available (3x current)
- Agent response tokens reduced by 50%

### After Wave 6
- Error recovery success rate >70% (from ~30%)
- Time to first successful task <2 min (from ~5 min)
- Event delivery latency <500ms (from 2000ms)

---

## Recommendation

**Start immediately with Wave 1 parallel tracks**:
1. Track A (Data Model) - Creates foundation for all container tracking
2. Track B (CLI Structure) - Improves immediate usability

This approach delivers:
- **Week 1-2**: Solid foundation (schema + CLI)
- **Week 3-4**: Container tracking online (prevents future incidents)
- **Week 5-6**: Full visibility (reconciler + feedback)
- **Week 7-9**: Complete introspection (CLI + MCP combined)
- **Week 10-11**: Real-time events
- **Week 12**: Polish

The $140 incident would be impossible after Wave 3 is complete.
