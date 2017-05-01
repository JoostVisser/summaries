# Process Algebra

[TOC]

$$
\texttt{LaTeX commands}
\newcommand{\N}{\mathbb N}
$$

## Lecture 0 - What is process algebra?

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

## Lecture 1 - Process Algebra

### Algebra of Natural Numbers

Consider set $\N$ together with three operations:

- $+ : \N \times \N \rightarrow \N$
- $\times : \N \times \N \rightarrow \N$
- $succ(n) = n+1$

#### Semantics

This is the Semantics of $(\N, +, \times, succ, 0)$:

- $n + 0  =n$
- $m + succ(n) = succ(n+m)$

These are some properties of the sets of operations.

#### Syntax / Logic

**Signature:** collection of symbols with an *arity*. Generally denoted with $\Sigma$.
Signature + variables determine 

**Equation:** Formula of the form $t=u$, where $t$ and $u$ are terms.
**Equational theory:** Pair $(\Sigma, E)$ consisting of a signature $\Sigma$ and set of $\Sigma$-equations $E$. 
These equations in $E$ are the **axioms** of the equational theory. 

Let $T=(\Sigma, E)$. 
We write $T \vdash t = u$ if $\exists$ derivation of $t=u$ using the following rules of equational logic:

- $\dfrac {}{t=u}$ $\leftarrow$ **Axiom**, if $t=u$ is an equation in $E$.
- $\dfrac {}{t=t} \leftarrow$ **Reflection**
- $\dfrac {t=u}{u=t} \leftarrow$ **Symmetry**
- $\dfrac{t=u \ \ \ \ \ \ u=v}{t=v} \leftarrow $ **Transitivity**
- **Substitution rule**
- **Cont**

In other words, you can completely formalize these derivations such that we can proof equations from a bunch of axioms.

We fine an interpretation $\iota$ of the function symbols as functions t





### Process

Behaviour is the execution of actions / events.

**Transition-system space** $(S, L, \rightarrow, \downarrow)$ consists of:

- Set $S$ of *states*.
- Set $L$ of *labels*.
- Transition relation: $\rightarrow \subseteq S \times L \times S$ 
- Set $\downarrow \subseteq S$ of *terminating* / *final* states.

Example:

- Bunch of states, like a finite automata. You can do some labels and eventually terminate.

#### Reachability

Reachability relation $\rightarrow^* \subseteq S \times L^* \times S$ is defined as:

1. $s \rightarrow^* s$ for all $s \in S$. 
2. ​

Basically means whether we can reach state $t$ from state $s$.

Transition system with root denotes all states that is reachable from state $r$.

#### Connection with Automata

A transition system is **regular** iff $S$ and $L$ are finite.
Regular transition system = finite automaton

A word $\sigma \in L^*$ is a complex execution / run of a transition 

Language: words that are recognized by the finite automaton. (Ways of going input $\rightarrow$ output.)