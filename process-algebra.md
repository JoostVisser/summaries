# Process Algebra

[TOC]

$$
\texttt{LaTeX commands}
$$

## Lecture 1 - What is process algebra?

### Small introduction

[Website](http://www.win.tue.nl/~luttik/Courses/PA/schedule.php)

**Reactive systems:** Systems that do not terminate.

**Concurrency theory:** Active field in CS that studies formalisms for modelling/analysing systems.

**Process Algebra:** Branch of concurrency theory.

1. Number of **atomic processes** $\longleftarrow$ Simplest behaviour
2. Define new **composition operators** $\longleftarrow$ More complex behaviours.

Consider the following example:
$$
(x:=1 || x:= 2) \cdot x := x + 2 \cdot (x:= x-1 || x:= x+5)
$$

- We don't know which parallel process will execute first, so there will be no unique output.
- The lowest output is 7, the largest output is 8.

### Contents of the course

- Various notions of composition.
  - This is ... . Examples: sequential composition / parallel composition.
- Expressiveness
- Various notions of behavioural equivalence.
- Axiom systems + quality.
- Abstraction.

