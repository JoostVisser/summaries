# Process Algebra - Lecturess

[TOC]

$$
\texttt{LaTeX commands}
\newcommand{\N}{\mathbb N}
\newcommand{\ra}{\rightarrow}
\newcommand{\la}{\leftarrow}
\newcommand{\lra}{\longrightarrow}
\newcommand{\lla}{\longleftarrow}
\newcommand{\bis}{\overset \leftrightarrow -}
$$

## Lecture 0 - What is process algebra?

### Small introduction

[Website](http://www.win.tue.nl/~luttik/Courses/PA/schedule.php)

**Classic view** of a computer program is a program that transforms an input form an output. Program $P$ is a partial function: $[[P]]:States \rightarrow States$, which always terminate.

What about vending machines? Operating systems? $\implies$**Reactive systems:** Systems that compute something by reacting to stimuli in the environment.

*Goals of this coarse*

- How to develop (design) a system that works
- How to analyse (verify) the design.

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
- $\dfrac{t=u \quad u=v}{t=v} \leftarrow $ **Transitivity**
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

## Lecture 2

### Some exercise stuff

To proof:
$$
T \vdash e(s^m(0), s^n(0)) = s^{m^n}(0)
$$
Proof: Induction on $n$

If $n=0$, then:
$$
\begin{align*}
T_2 \vdash e(s^m(0), s^n(0))
&\equiv e(s^m(0), 0) \\
& = s(0) &[PA5] \\
& \equiv s^{m^n}(0)
\end{align*}
$$
Let $n\geq 0$ and suppose $T_2 \vdash e(s^m(0), s^n(0)) = s^{m^n}(0)$. (Induction Hypothesis)

Then:
$$
\begin{align*}
T_2 \vdash e(s^m(0),s^{n+1}(0))
&\equiv e(s^m(0), s(s^n(0)) \\
& = m(e(s^m(0), s^n(0)), s^m(0)) & [PA6]\\
& \overset {IH} = m(s^{m^n}(0), s^m(0)) \\
& = s^{m^n m }(0) & \text{[By (2.2.3)]}\\
& \equiv s^{m^{n+1}}(0)
\end{align*}
$$
### Lecture part

Binary relation $R$ on the set of states $S$ of a transition-system space is a **bisimulation relation** iff $\forall{s ,t \in S}$ where $s\ R\ t$:

1. If $s \overset a \lra s' ​$ for some $a \in L​$ and $s' \in S​$, then $\exists t' \in S​$ such that $t \overset a \lra t'​$ and $s'\ R\ t'​$
2. ​

$a \bis b$ 



## Lecture 3

### Missed the first part

### Second part

He