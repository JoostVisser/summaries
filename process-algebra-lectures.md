# Process Algebra - Lectures

[TOC]

$$
\texttt{LaTeX commands}
\newcommand{\N}{\mathbb N}
\newcommand{\ra}{\rightarrow}
\newcommand{\la}{\leftarrow}
\newcommand{\lra}{\longrightarrow}
\newcommand{\lla}{\longleftarrow}
\newcommand{\bis}{\overset \leftrightarrow -}
\newcommand{\0}{\mathbf 0}
\newcommand{\s}{\mathbf s}
\newcommand{\a}{\mathbf a}
\newcommand{\m}{\mathbf m}
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

## Lecture 4

| Logic                                    | Semantics                                |
| ---------------------------------------- | ---------------------------------------- |
| Formalising reasoning about things       | Mathematical representation of 'real world' phenomenon. |
| Signature $\Sigma = (\mathbf a, \mathbf m, \mathbf s, \mathbf 0))$ | $(\mathbf N, +, \times, succ, 0) \leftarrow$ functions on natural numbers |
| Term $\a(\0,\s(\0))$                     |                                          |
| Axioms $\a(x,\0)=x$                      |                                          |

Now we can do a similar thing for processes:

| Logic                   | Semantics                          |
| ----------------------- | ---------------------------------- |
| Signature (0, a._, +)   | Operational semantics (Operations) |
| Terms (a.0+b.(c.0+d.0)) | Transition system space            |
|                         | Bisimilarity                       |
|                         | Also consists of an algebra.       |
|                         | Equivalence classes                |

If $p\bis q$, then $a.b \bis a.q$

We link this via the interpretation, that links symbols and operations:

- $0 \mapsto \0$
- $+ \mapsto \mathbf +$
- $a. \mapsto \a.$

Example of interpretation:
$$
\begin{align*}
\iota_\alpha(a.0 + b.c.0)
&= \iota_\alpha(a.0)\iota(+)+\iota(b.c.0) \\
&= \a.\iota(0)\iota(+)+\iota(b.c.0) \\
\end{align*}
$$
How to proof that an axiom is complete?

Consider the axiom $x+y = y+x$.

To show that it is valid / soundness in $\mathbb P(MPT(A)) /_\bis$ we should prove that $\iota_\alpha(x+y) = \iota_\alpha(y+x)$. We can reduce it to if $[p+q]_\bis$ is equal to $[q+p]_\bis$, thus we should prove that $p+q$ is bisimilar to $q+p$, which is done in the previous lemma.



How to proof that an axiom is ground-complete?

An algebra $\mathbb A$ is *ground-complete* for the algebra ... if ... $\vDash p=q \implies MPT(A) \vdash p=q$.

After some notational stuff, we have to show that $p \bis q \implies p = q$.
Then again, how would you proof that $p=q$? We should apply the axioms.

So given that two terms are bisimilar, we show equality via the four axioms of $MPT(A)$.



Initial algebra $\mathbb I(\Sigma_1, E_1)$ is all the terms but with their equivalence classes, i.e. $[\mathbf 0]_{=}$. What we're basically proving with ground-completeness and soundness is that this class is isomorph with the algebra, which basically means that there is a one-on-one mapping between these two classes.

#### Part 2

Proof idea for ground completion: 

1. Assume that $q = a_1.q_1 + \cdots + a_k.q_k$, involving A1, A2, A3 and A6.





How to proof for all closed MPT(A)-terms that $p \bis p + (q_1 + q_2) \implies p \bis p+q_1 \wedge p \bis p+q_2$.

> Proof
>
> We assume that $p \bis p+(q_1 + q_2)$
>
> Let R be a bisimulation relation such that $p\ R\ [p + (q_1 + q_2)]$.
>
> Define $R' = R \cup \left\{(s, s+t_1) | \exists_{t_2} s\ R\ [s + (t_1 + t_2)]\right\} \cup \{(s, s) | s \in \mathcal C(MPT(A)\}$
>
> Then 

#### Part 3

Extension to MSP: TSP - Sequential composition.

- We've completely ignored termination behaviour. We want to add a constant for successful termination. 

Consider $a.1 \cdot b.1$ first executes $a$ and then execute action $b$.
Consider $a.0 \cdot b.1$ first executes an $a$ but then stops because it's in a deadlock.
Consider $(a.1 + b.1) \cdot (a.1 + b.1)$, we can first execute $a$ or $b$, then we subsequently execute again and we can execute $a$ or $b$ again.
Consider $(a.1 + 1) \cdot b.1$, then we can do an $a$ step and successfully terminate, then a $b$ step, or it can directly do a $b$ step. 

Marc says: ":haircut::halloween::halt::japanese_ogre::eight_pointed_black_star:". Joost did not reply to this nonsense.

## Lecture 5

### **Exercise 4.2.1**

Derive A2' from $MPT(A)$.
$$
\begin{align*}
MPT(A) \vdash (x+y) + z 
& = x+(y+z) & [A2] \\
& = (y+z)+x & [A1]
\end{align*}
$$
But we should also do it the other way around, deriving $A1$ from $A2'$ and $A3$.
$$
\begin{align*}
MPT(A) \vdash x+y 
&= (x+x) + y & [A3] \\
&= (x+y)+x & [A2'] \\
&= (y+x)+x & [A2'] \\
&= ((y+y)+x) + x & [A3] \\
&= ((y+x)+y)+x & [A2'] \\
&= (y+x) + (y+x) & [A2'] \\
&= y+x
\end{align*}
$$
We can also simplify the proof by applying the axioms the other direction.

### Basic Sequential Processes

The theory of $BSP(A)$ is the same as $MSP(A)$ but with th added constant 1 for **successful termination**.

We can finally use our termination predicate $\downarrow$, yay! :D

**SOS meta-theory** - Determining that bisimilarity is a congruence to these TSP.

- A collection of *operational rules* is in **path format** iff every rule in $R$ holds:
  - The target of the premise (transitions above the line) only contains a single variable.
  - Source of the conclusion (transition below the line) should either be:
    - Single variables
    - A function symbol applied to variables.
  - The variables in the source of the conclusion + target of premise and see whether one of these variables occurs more than once in either of these positions.

If the operational rules are all in the path format, then *bisimilarity* is a congruence for that process calculus.

For all closed $BSP(A)$-terms, $BSP(A) \vdash p=1 \Longleftrightarrow p \bis q$.

### Sequential Processes

We extend $BSP(A)$ with a binary operator $\cdot$ for sequential composition:

- Process $p \cdot q$ executes $p$ first and, upon successful termination of $p$, then executes $q$.
- $(a.1+ b.1) \cdot (a.1 + b.1)$ first executes either an $a$ or a $b$ and again either an $a$ or a $b$.



We have some extra operational semantics.

$a.1 \overset a \rightarrow 1$
$a.1\cdot b.1 \overset a \rightarrow 1 \cdot b.1$

Now we still have $1 \cdot b.1$ over. We can use this rule to indicate that:

$1 \cdot b.1 \overset b \rightarrow 1$

Remember that $\downarrow$ is the least set satisfying the rules above. If term $p \in \downarrow$. So apparently if this is true, it has to be so according to the roles, as $\downarrow$ is the *least* set that adheres to the rules, this means that there *has to be* a proof that $p$ has to terminate.

Why can $a.1$ not be terminating? We don't have any possibility in our operational semantics that $a.1\downarrow$ does not hold. By this reasoning we can conclude that $a.1 \not \downarrow$. 

Then my questions are:

1. Can we express in $TSP(A)$ more behaviour than in $BSP(A)$?
2. Does the operational semantics for $TSP(A)$ change the behaviour of $BSP(A)$-terms?
   - In other words, is it *operationally conservative extension*? 
     Do these three more rules influence behaviour of the old terms, such as $a.1 + b.0$?
     1. No, since the sources of the conclusions of all operational rules not stemming from $BSP(A)$ terms are not $BSP(A)$-terms.
     2. No, the operational rules stemming from $BSP(A)$ are *source-dependent*, i.e. we cannot introduce new arbitrary terms, such as $TSP(A)$ terms.
     3. Therefore, according to Theorem 3.2.19, $TSP(A)$ is an *operationally conservative extension*, i.e. $p \bis q$ in $TSP(A) \Longleftrightarrow p \bis q$ in $BSP(A)$.

We want to create a strategy to eliminate our new behaviour by removing every term with sequential composition to without sequential composition.

#### Elimination

For every closed *BSP(A)*-term $p$ it holds that for every closed $BSP(A)$-term there exists a closed $BSP(A)$-term $r$ such that $p \cdot q$ = $r$.

> If $p \equiv 0$, then by A7, $p \cdot q = 0$, which is a $BSP(A)$-term.
>
> If $p \equiv 1$, then by A9, $p \cdot q = q$, which is a $BSP(A)$-term.
>
> Suppose that $p \equiv a.p'$ and sassume that for every BSP(A)-term there exists a BSP(A)-term r' such that $p' \cdot q = r$ (IH)
>
> ​	Then, by A10 and IH, $p \cdot q = a.(p' \cdot q) = a.r'$, which is a $BSP(A)$-term.
>
> ​	Suppose that $p \equiv p_1 + p_2$, and suppose ...
>
> Now for every arbitrary $TSP(A)​$-term $p​$ there exists a closed $BSP(A)​$-term q such that $p=q​$.





