---
layout: post
title: "Monadris: Why Functional Design Makes Tetris Safer and Easier (Scala 3 + ZIO)"
description: "Separating a pure core from an impure ZIO shell — a deterministic state machine, ADTs, a serialized event loop, and replay — makes a Tetris implementation safer and easier to evolve."
date: 2026-02-06
lang: en
permalink: /en/monadris-functional-design/
---

Tetris looks simple — until you implement it.

A real Tetris loop has time (ticks), concurrent inputs (keystrokes), state transitions (collision, locking, line clears), and non-determinism (piece generation). In many imperative designs, these concerns end up tangled in shared mutable state, which tends to produce bugs that are:

- hard to reproduce (timing-dependent),
- hard to test (logic mixed with effects),
- hard to debug (replay isn’t deterministic).

**Monadris** treats those failure modes as architectural problems, not “be more careful” problems. The core idea is to **physically separate** a *pure* core (domain + rules) from an *impure* app layer (I/O + timing + rendering), and to model the game as a deterministic state machine driven by events.

<figure>
  <img src="{{ '/assets/img/architecture-core-shell.svg' | relative_url }}" alt="A pure core, (State, Input) to State, enclosed by an impure ZIO shell handling I/O, timing, rendering, and persistence">
  <figcaption>A <em>pure</em> core sits inside an <em>impure</em> shell: events flow in, the next state flows out, and the boundary is enforced at compile time.</figcaption>
</figure>

## What “safer” means here

When I say “functional design makes Tetris safer,” I don’t mean “FP is morally superior.” I mean the architecture makes certain classes of bugs harder (or impossible) to introduce:

- No shared mutable game state across concurrent input/timer logic
- Single, serialized state transition path (events in, state out)
- Core rules are deterministic and testable without any runtime, clock, or terminal
- Replay can be made deterministic by recording the right events

## Why imperative Tetris often breaks in practice

Here’s a common failure mode when ticks and inputs race on shared mutable state:

- At time *t*, the player presses MoveDown
- At nearly the same time, the Tick arrives
- Both handlers read the same state (same falling piece position)
- Both apply “move down” and write back

Depending on scheduling, you can get:

- a double-step fall,
- a collision check on a stale position,
- a lock/clear occurring “too early,”
- a bug that disappears under debugging (classic heisenbug).

<figure>
  <img src="{{ '/assets/img/race-condition.svg' | relative_url }}" alt="A timer Tick and a MoveDown input both read and write the same shared mutable state, causing a race">
  <figcaption>The imperative failure mode: a tick and an input both read and write the same mutable state, racing to produce a double-step, a stale-position collision, or a heisenbug.</figcaption>
</figure>

Monadris doesn’t try to “time it right.” It **defines a single ordering**: events are queued, then processed one-by-one by the game loop.

## 1. Freeze the core as a pure state machine
At the center of the design is a strict rule:

> The core game logic is just a function from (State, Input) to State.

This is explicitly called out as a first-class feature of the project.

In practice, it means the core never:

- reads the clock,
- polls input devices,
- writes to the terminal,
- talks to files or databases.

Those things happen elsewhere.

Once you lock in that boundary, the core becomes easy to reason about:

- same state + same input → same next state
- no “hidden” sources of state (time, randomness, I/O)

This is the foundation for both reliable tests and reliable replay.

## 2. Make invalid states and forgotten branches harder with ADTs
In a game loop, “inputs” and “statuses” tend to balloon over time (pause, quit, soft drop, hard drop, rotations, etc.). Encoding those as algebraic data types (Scala 3 enum) pays off because pattern matches become exhaustive.

The architectural goal is simple:

- add a new input variant → the compiler guides you to the places you must handle it

This isn’t about style. It’s about controlling the surface area of change as the project evolves.

## 3. Enforce the purity boundary as a team constraint, not a convention

Monadris doesn’t rely on “developers will be disciplined.” It makes certain shortcuts expensive by enforcing a “no mutation” posture in the pure core:

- no var
- no null
- no throw
- no return

…and it does this *at compile time* in the core module.

Important nuance: this doesn’t “prove” mathematical purity of everything on the JVM. What it does do is **protect the architectural boundary** from drifting into “just one exception here” or “quick mutable cache there,” which is how clean designs usually decay over time.

## 4. Concurrency at the edges, serialization at the state transition
Inputs and ticks are inherently concurrent. The design choice is:

- allow concurrency only in **event production**,
- keep the **state transition** single-threaded and deterministic.

Conceptually:

1. input fiber pushes events into a queue
2. timer fiber pushes tick events into the same queue
3. one game loop consumes events and applies update(state, input) sequentially

<figure>
  <img src="{{ '/assets/img/event-loop.svg' | relative_url }}" alt="Input and timer fibers push events into one queue; a single game loop consumes them one at a time and applies update(state, input) to produce the next state">
  <figcaption>Concurrency lives only in the producers. Events funnel through one queue into a single game-loop consumer, so there is exactly one serialized path to the next state.</figcaption>
</figure>

That last point is the “safety lever”: there is exactly one path that produces the next state.

## 5. Deterministic replay: recording inputs is not enough
A common replay mistake is to log only the user’s inputs. That fails as soon as your game contains non-determinism (e.g., next piece generation).

Deterministic replay works when:

> “events in” are sufficient to reconstruct “state out.”

You can do this in two broad ways:

- make randomness deterministic (seeded RNG threaded through state), or
- **record the non-deterministic outcomes** (e.g., which piece actually spawned) as events, and feed them back during replay.

## 6. Testing becomes direct state-transition testing
Once the core is (State, Input) => State, tests stop being “integration-ish” and become truly unit-level:

- build an initial state
- apply an input
- assert on the new state

No clock mocking, no terminal harnesses, no asynchronous flakiness.

This is especially valuable for:

- collision detection
- line clearing
- scoring / level transitions
- game-over conditions
- replay reconstruction

## Why ZIO (briefly)

ZIO is a natural fit for this architecture because it makes “impure shell” concerns explicit and composable:

- **Fibers** for structured concurrency (inputs + ticks as concurrent producers)
- **Queue** for safe, non-blocking event buffering
- clear separation between **pure functions** and **effects**
- testing support for effectful boundaries without contaminating the core

The result is: concurrency stays at the edge, while the core remains deterministic.

## Summary: it’s not “FP makes it safe” — the boundaries do
Monadris is “functional” in the specific architectural sense that matters:

- **Pure core**: deterministic rules and transformations
- **Impure shell**: ZIO effects for input, time, rendering, persistence
- **Event loop**: concurrent producers, serialized consumer
- **Replay**: record what’s needed to reconstruct state deterministically
- **Guardrails**: compile-time constraints that keep the core pure over time

These boundaries are what make the system safer and easier to evolve.

## Links (stable, code-oriented)

- [Repository](https://github.com/fffclaypool/monadris)
- [GameLogic.update()](https://github.com/fffclaypool/monadris/blob/v1.1.0/core/src/main/scala/monadris/game/GameLogic.scala#L13-L34)
- [Event loop / queue consumption](https://github.com/fffclaypool/monadris/blob/v1.1.0/app/src/main/scala/monadris/infrastructure/game/GameLoopRunner.scala#L54-L69)
- [Replay events (SpawnEvent / replay model)](https://github.com/fffclaypool/monadris/blob/v1.1.0/core/src/main/scala/monadris/replay/ReplayData.scala#L6-L8)
