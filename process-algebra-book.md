# Process Algebra - Book summary

[TOC]

$$
\texttt{LaTeX commands}
\newcommand{\B}{\mathcal B}
\newcommand{\C}{\mathcal C}
\newcommand{\T}{\mathcal T}
\newcommand{\0}{\mathbf 0}
\newcommand{\a}{\mathbf a}
\newcommand{\m}{\mathbf m}
\newcommand{\s}{\mathbf s}
\newcommand{\bis}{\overset \leftrightarrow -}
$$

## Chapter 2 - Preliminaries

### Equational theory

#### What is the formal definition of equational theory and terms?

**Equational theory:** Signature + set of equations over this signature (basic laws)

- **Signature** $\Sigma$ is a set of constant symbols + function symbols with their arities. 
  - *Arity* of a function is the number of arguments the function takes.
  - Example: $\Sigma_1$ contains the constant **0**, unary function **s** for successor and binary function symbols **a** and **m** for addition and multiplication respectively.

Set of all **terms** over $\Sigma$ and a set of variables $V$, denoted as $\mathcal T(\Sigma, V)$, is the smallest set such that:

1. Each variable in $V$ is a term in $\mathcal T(\Sigma, V)$.
2. Each constant in $\Sigma$ is a term in $\mathcal T(\Sigma, V)$.
3. For each $n$-ary function symbols where $t_1, \ldots, t_n$ are terms in $\mathcal T(\Sigma, V)$, then the result of the function is a term in $\mathcal T(\Sigma, V)$.

Basically, all possible values that can be derived from the variables and symbols are in $\mathcal T(\Sigma, V)$.
Shorthand notations: $\mathcal T(\Sigma, V) = \mathcal T(\Sigma) = $ $\Sigma$-terms or even *open* terms (can contain variables).

- **Closed term (ground term):** Term that does not contain variables.
- $\mathcal C(\Sigma)$ - Set of all closed terms over signature $\Sigma$.

**Equational theory:** Tuple $(\Sigma, E)$ where:

- $\Sigma$ is a signature 
- $E$ is a set of equations in the form of $s=t$, where $(s,t \in \mathcal T(\Sigma))$. Also known as **axioms**.

Goal of equational theory: find which terms over $\Sigma$ of this theory are considered equal.

#### Substituting a variable

**Substitution:** Substitute arbitrary terms for these variables.

- Substitution $\sigma:$ $V \rightarrow  \mathcal T(\Sigma, V)$, where for any term $t$ in $\mathcal T(\Sigma, V)$:
- $t[\sigma]$ denotes the term obtained by *simultaneously* substituting *all* variables in $t$ according to $\sigma$.
  - For each variable $x$: $x[\sigma] = \sigma(x)$. 
  - For each constant $c$: $c[\sigma] = c$.
  - For each $n$-ary function symbol $f$: $f(t_1, \ldots, t_n)[\sigma]$ is the term $f(t_1[\sigma], \ldots, t_n[\sigma])$.
- In words: this is just the "definition" of variables where we can put any term inside this variable. We can also substitute variables for variables, as these are also terms.
- Example: consider the term $t \equiv \mathbf a(\mathbf m(x,y),x)$. Let $\sigma_1$ be a substitution mapping as follows: $x \rightarrow \mathbf s (\mathbf s(\mathbf 0))$ and $y \rightarrow \mathbf 0$. Then $t[\sigma_1] \equiv \mathbf a(\mathbf m( \mathbf s (\mathbf s(\mathbf 0)),\mathbf 0), \mathbf s (\mathbf s(\mathbf 0)))$ 

#### Derivability

How do we derive other equations from our current set of equations? There are a standard set of proof rules that help us with this:
Equation $s=t$ is derivable from theory $t$, denoted as $T \vdash s=t$, iff it follows from the following rules:

1. **Axiom rule:** For any equation $s=t \in E \implies T \vdash s=t$.
   I.e. all rules in $E$ are already derived.
2. **Substitution rule:** For any terms $s, t$ and substitution $s$: $T\vdash s=t \implies T\vdash s[\sigma] = t[\sigma]$.
   I.e. we can substitute variables to derive new equations.
3. **Reflexivity:** For any term $t \in \mathcal T(\sigma)$: $T \vdash t = t$.
   I.e. any term implies itself.
4. **Symmetry:**  For any terms $s, t$: $T\vdash s=t \implies T\vdash t=s$.
   I.e. equality goes two directions.
5. **Transitivity:** For any terms $s, t, u$: $T \vdash s=t \cup T \vdash t=u \implies T \vdash s=u$
   I.e. transitivity: equality holds over multiple terms.
6. **Context rule:** For any $n$-ary function symbol $f$, we have that:
   $T \vdash t_i = s \implies T\vdash f(t_1, \ldots, t_n = f(t_1, \ldots, t_{i-1}, s, t_{i+1}, \ldots, t_n)$
   I.e. we can put a function around the left and r.h.s. putting $s$ in the place of $t_i$.

With these rules, we can derive different equations via deviation from the axioms.

#### Proving by natural induction

We generally use natural induction as the standard technique to proof equational theories. Consider the example of addition, succession and multiplication, then we can write all formulas as the following **basic terms**, which is the smallest set of terms possible $\mathcal B (\Sigma_1)$:

1. $\0 \in \B(\Sigma_1)$
2. $p \in \B(\Sigma_1) \implies \s(p) \in \B(\Sigma_1)$

This can be rewritten as follows, for ease of notation:

1. $\mathbb s^0(\mathbb 0) = \mathbb 0$
2. $\mathbb s^{n+1}(\mathbb 0) = \mathbb s(\mathbb s^n(\mathbb 0))$

Using these rules, we can create an equation with addition:

- For any $p,q \in \mathcal B(\Sigma_1)$ with $p=s^m(0)$ and $q=s^n(0)$:
  $T_1 \vdash a(p,q) = s^{m+n}(0)$

Proof:

- **Base case** - Assume $n=0$, $T_1 \vdash a(p,q) = a(s^m(0),s^0(0)) = a(s^m(0),0)=s^m(0) = s^{m+n}(0)$.
- **Inductive case** - Assume it holds for all $n \leq k$ for a certain $k$. Then we have that:

$$
\begin{align*}
T_1 \vdash a(p,q) 
&= a(s^m(0), s^{k+1}(0)) \\
&= a(s^m(0), s(s^{k}(0))) & [\text{Definition 2.2.1}]\\
&= s(a(s^m(0), s^{k}(0))) & [\text{Axiom PA2}] \\
&= s(s^{m+k}(0)) & [\text{IH}]\\
&= s^{m+k+1}(0) & [\text{Definition 2.2.1}]
\end{align*}
$$

#### Proving by structural induction

Another form of induction is **structural induction**. Let $P(t)$ be some property on terms over $\Sigma$:

- **Base cases:** For any constant symbol $c$ in $\Sigma$: $P(c)$ holds.
- **Inductive steps:** For any $n$-ary function symbol $f$ in $\Sigma$ and closed terms $p_1, \ldots, p_n \in \C(\Sigma)$:
  Assume that $\forall_{1 \leq i \leq n}: P(p_i)$ holds $\implies$ $P(f(p_1, \ldots, p_n))$ holds.
- This is sort of an induction over functions.

Then $P(p)$ holds for any closed $\Sigma$-term $p \in \C(\Sigma)$.

**Elimination:** Any closed $\Sigma_1$-term ($\C(\Sigma_1)$) can be written as *basic* $\Sigma_1$-terms $(\B(\Sigma_1))$.

- In terms of our example - Since any term is a combination of $\s$ function symbols with $\0$ constant symbols, we can eliminate $\a$ and $\m$ from any closed term.

To proof: For any $p​$ in $\C(\Sigma_1)​$, there exists some $p_1 \in \B(\Sigma_1)​$ such that $T_1 \vdash p = p_1​$.

> Proof by structural induction
>
> **Base case** 
> Assume that $p \equiv 0$. Since $p$ is a basic term, this is trivially satisfied for $p_1 = p$.
>
> **Inductive steps**
>
> 1. Assume $p \equiv s(q)$ for some closed term $q \in \C(\Sigma_1)$. By induction, we can assume that there is some basic term $q_1 \in \B(\Sigma_1)$ such that $T_1 \vdash q = q_1$. We know that $T_1 \vdash s(q) = s(q_1)$ and since $s(q_1) \in \B(\Sigma_1)$ by definition of the basic terms, this property holds.
> 2. Assume $p \equiv \a(q,r)$ for some closed term $q \in \C(\Sigma_1)$. By induction, let $q_1, r_1 \in \B(\Sigma_1)$ be the basic terms such that $T_1 \vdash q = q_1$ and $T_1 \vdash r = r_1$. Then, let us rewrite $q_1 = s^m(\0)$ and $q_2 = s^n(\0)$ with $n, m \in \mathbb N$. By (2.2.2), it follows that:
>    $T_1 \vdash p = \a(q, r) = \a(\s^m(\0), \s^n(\0)) = \s^{m+n}(\0) $ which is a basic $\Sigma_1$ term.
> 3. Assume $p \equiv \m(q,r)$ for some closed term $q \in \C(\Sigma_1)$. By induction, let $q_1, r_1 \in \B(\Sigma_1)$ be the basic terms such that $T_1 \vdash q = q_1$ and $T_1 \vdash r = r_1$. Then, let us rewrite $q_1 = s^m(\0)$ and $q_2 = s^n(\0)$ with $n, m \in \mathbb N$. By (2.2.3), it follows that:
>    $T_1 \vdash p = \m(q, r) = \m(\s^m(\0), \s^n(\0)) = \s^{mn}(\0) $ which is a basic $\Sigma_1$ term.

#### Extension of an equational theory

Theory $T_2 = (\Sigma_2, E_2)$ is an **extension** of theory $T_1 = (\Sigma_1, E_1)$ $\Longleftrightarrow$ $\Sigma_1 \subseteq \Sigma_2$ and $E_1 \subseteq E_2$.

- For example, adding function symbol $\mathbf e$ for exponentiation to Theory $T_1$.

Theory $T_2$ is a **conservative extension** of $T_1$ if it is an extension as well as the following holds:

- All $\Sigma_1$-terms $s$ and $t$, $T_2 \vdash s=t \implies T_1 \vdash s=t$.
- Adding exponentiation to Theory $T_1$ is a *conservative extension* of $T_1$.

Theory $T_2$ is a **ground-extension** of theory $T_1$ iff $\Sigma_1 \subseteq \Sigma_2$ and for all closed $\Sigma_1$-terms $p$ and $q$, $T_1 \vdash p=q \implies T_2 \vdash p=q$.

- Is similar but weaker version of extension, only holding for closed terms in $\Sigma_1$.
- $T_2$ is a **conservative ground-extension** if it is a ground-extension and the implication is a bi-implication, i.e. for all $\Sigma_1$-terms $s$ and $t$, $T_2 \vdash p=q \implies T_1 \vdash p=q$.

Any closed $\Sigma_2$-term of the exponential $T_2$ can be written as a basic $\Sigma_1$ term as well!

We can use these proofs whenever one is asking for a proof on closed terms. Since any closed term can be rewritten as basic terms, we just have to proof it for basic terms.

### Algebra

An **Algebra** $\mathbb A$ consists of:

- Elements of $\mathbf A$
- Constants in $\mathbf A$
- Functions over $\mathbf A$

The set of elements $\mathbf A$ of $\mathbb A$ is called the *universe*, *domain* or *carrier set* of $\mathbb A$.

Example - Algebra $\mathbb B = (\mathbf B, \wedge, \neg, true)$ is the algebra of:

- Booleans $\mathbf B = \{true, false\}$
- $\neg$ is a unary function, $\wedge$ is the binary function (conjunction) on Booleans
- $true$ is a constant.

Example 2 - Algebra $\mathbb N = (\mathbf N, +, \times, succ, \0)$.

An algebra $\mathbb A$ is a **$\Sigma$-algebra** if there exists a mapping from symbols of $\Sigma$ into constants and functions of the algebra of $\mathbb A$. This mapping is called an *interpretation*.

Example of the $\Sigma_1$ signature. Interpretation $\iota_1$ is:

- $\0 \mapsto 0$
- $\s \mapsto succ$
- $\a \mapsto +$
- $\m \mapsto \times$

We can for example give another interpretation $\iota_2$ to these symbols, where $\a \mapsto \times$ and $\m \mapsto +$.y1

What is the difference between all those symbols of the *signature* and the real constants and functions from the *algebra*? 

- The symbols contains merely symbols without meaning.
- The algebra gives them meaning by mapping them to actual constants and functions.

#### Validity

Let $\Sigma$ be a signature, $V$ a set of variables and $\mathbf A$ a $\Sigma$-algebra with domain $\mathbf A$ and $\iota$ an interpretation of $\Sigma$ onto $\mathbf A$. 

Given open term $t \in \T(\Sigma,V)$, $\iota_\alpha(t)$ is the elements of $\mathbf A$ obtained by:

- Replacing all symbols in $t$ with constants + functions of $\mathbf A$ according to $\iota$ 
  For each constant $c \in \Sigma : \iota_\alpha(c) = \iota(c)$ 
  For each $n$-ary function: $\iota_\alpha(f(t_1, \ldots, t_n)) = \iota(f)(\iota_\alpha(t_1), \ldots, \iota_\alpha(t_n))$
- Replacing all variables by elements of $\mathbf A$ according to $\alpha$.
  For all variables $x \in V: \iota_\alpha(x) = \alpha(x)$. 

Equation $s=t$ is valid in $\mathbb A$ under interpretation $\iota$, denoted as $\iota \vDash s=t$, $\Longleftrightarrow \iota_\alpha(s) =_\mathbf A \iota_\alpha(t)$, where $=_\mathbf A$ is the identity on domain $\mathbf A$.

Example - Given $\iota_1$ and $\iota_2$ as defined in the previous chapter and algebra $\mathbb N$. Then:

- Equation $\a(x,y) = \a(y,x)$ is valid in $\mathbf N$ under interpretation $\iota_1 \vDash \a(x,y) = \a(y,x)$ and $\iota_2 \vDash \a(x,y) = \a(y,x)$, since both $m+n =_\mathbf N n+m$ and $m \times n =_\mathbf N n\times m$ holds for $m,n \in \mathbf N$.
- Equation $\a(x,\0)=\0$ is valid under $\iota_1$, as $n+0 = _\mathbf N n$, but not for interpretation $\iota_2$, as $n \times 0 \neq n$.

#### Model

An algebra $\mathbb A$ is a *model* of $T$ w.r.t. $\iota$, denoted as $\mathbb A, \iota \vDash T$ (or with $E$ instead of $T$), iff for all equations $s=t \in E$ it can be derived from $\mathbb A$, i.e. $\mathbb A, \iota \vDash s=t$.
If $\mathbb A$ is a model of $T$, then $T$ is a *sound axiomatization* of $\mathbb A$.
In words, all axioms of $T$ are valid in $\mathbb A$ under interpretation $\iota$. 

**Soundness:** For all $\Sigma$-terms $s$ and $t$: $T \vdash s=t \implies \mathbb A, \iota \vDash s=t$.

Example: 

- Algebra $\mathbb N$ under interpretation $\iota_1$ is a model of $T_1$, but not under interpretation $\iota_2$.
- We can even create a boolean interpretation with xor to be a model of $T_1$ (2.3.11).
  In order to proof this, we need to show that all axioms are valid in this algebra.

**Extension:** Let $T_2$ be an extension of $T_1$. Then if $\mathbb A$ is a model of $T_2 \implies \mathbb A$ is a model of $T_1$. 

**Equivalence classes:** Let $U$ be the universe of some elements and $\sim$ be an equivalence relation of $U$. Then the *equivalence class* the set of all elements equivalent to that element in $U$.
$[u]_\sim = \{v \in U | u \sim v\}$, here $u$ is representative of this class.

**Congruence:** Let $\mathbb A$ be an algebra with domain $\mathbf A$. Binary relation $\sim$ is a *congruence* on $\mathbb A$ iff:

1. $\sim$ is an equivalence on $\mathbf A$.
2. For every $n$-ary function $f$ and any elements in $\mathbf A$: 
   $a_1 \sim b_1, \ldots, a_n \sim b_n \implies f(a_1, \ldots, a_n) \sim f(b_1, \ldots, b_n)$  

**Quotient algebra:** The *quotient* algebra $\mathbb A$ *modulo* $\sim$, denoted as $\mathbb A_{\setminus \sim}$, has a universe of the equivalence classes of $\mathbb A$ under $\sim$ where all constants and functions are their respective equivalence classes.

$[p]_\vdash$ are all the terms in $T$ that are derivably equal to $p$.

**Completeness:** $T$ is complete for $\mathbb A$ iff for all *open* $\Sigma$-terms $s$ and $t$, $A,\iota \vDash s=t \implies T\vdash s=t$
**Ground-completeness:** $T$ is complete for $\mathbb A$ iff for all *closed* $\Sigma$-terms $p$ and $q$, $A,\iota \vDash p=q \implies T\vdash p=q$

**Isomorphisms of algebras:** Let $\mathbb A_1$ and $\mathbb A_2$ be two algebras with domain $\mathbf A_1$ and $\mathbf A_2$ respectively. These are *isomorphic* iff there exists a *bijective function* $\phi: \mathbf A_1 \rightarrow \mathbf A_2$ and another bijective function that maps the constants + functions of $\mathbb A_1$ onto those of $\mathbb A_2$.

## Chapter 3 - Transition Systems

### What are transition systems?

The goal of this coarse is to model reactive systems. To describe these systems, *automata* / *transition system* have been chosen. Automatas:

- Model systems in terms states + transitions which leads from one state to the next.
- Describes the operational behaviour of a system.

**Transition-system space:** The space $(S, L, \rightarrow, \downarrow)$ consists of labels $L$, states $S$, one 3-ary relation $\rightarrow$ and a subset of final states $\downarrow$:

- $\rightarrow \subseteq S \times L \times S$ is the set of transitions.
  - $s \overset a \rightarrow t$ is used for $(s,a,t) \in \rightarrow$, i.e. $s$ has an $a$-step to $t$. Reverse holds for $s \overset a {\not \rightarrow} t$.
- $\downarrow \subseteq S$ is the set of *terminating* / *final* states.
  - $s\downarrow$ indicates that $s \in \downarrow$, i.e. $s$ has a termination option. Reverse holds for $s \not \downarrow$

**Reachability:** all states that can be reached from a certain state via transitions:

- $\rightarrow^* \subseteq S \times L^* \times S$
- State $t \in S$ is *reachable* from $s \in S$ iff exists a word $\sigma \in L^*$ such that $s \overset \sigma {\rightarrow^*} t$.

We can induce a **transition system** from every state $s \in S$ which consists of all states reachable from $s$. State $s$ is called the *initial state* or *root*.

A **word** $\sigma \in L^*$ is a *complete execution* or *run* iff $\exists t \in S$ such that $s \overset \sigma {\rightarrow^*} t$ and $t \downarrow$.

- Two transition systems are language equivalent iff they have the same set of *runs*.
  However, sometimes language equivalent is not enough (lady & tiger example).

### Bisimulation relation

**Bisimilarity:** A binary relation $R$ on the set of states $S$ is a bisimulation relation iff:

1. For all states $s, t, s' \in s$, whenever $(s, t) \in R$ and $s \overset a \rightarrow s'$, then there is a state $t'$ such that $t \overset a \rightarrow t'$ and $(s',t') \in R$. Same holds vice versa 
2. Whenever $(s, t) \in R$ and $s \downarrow$ then $t \downarrow$ and vice versa.

These conditions are also called the transfer conditions. A shorthand notation of two states that are bisimilar is $s\ R\ t$.

**Bisimulation equivalent:** Two transition systems $s$ and $t$ are *bisimilar* iff there exists any bisimulation relation $R$ such that $s\ R\ t$. This is denoted as $s \bis t$.

One can consider this as playing a game with two players, where the second player can mimic the exact moves of the first player and vice versa. If there is any move that one player can do which the other player cannot, then these transition systems are not bisimilar.

---

**Theorem 3.1.13 - Equivalence**

Bisimilarity is an equivalence.

----

**Deadlock:** state $s$ is a *deadlock state* iff it does not have any outgoing transitions and is not a final state. 

A transition system is deadlock-free if it doesn't have a state with a deadlock.

A transition system is **regular** iff $S$ and $\rightarrow$ are finite. (Finite automaton)

Transition system is *finitely branching* iff each of its states has only finitely many outgoing transitions. 

## Chapter 4 - Basic process theory

Process theories: equational theories about reactive systems.

### The process theory MPT

**MPT(A):** Minimal Process Theory, where $A$ is the set of actions, a parameter of this theory.
Terms over the signature of this theory specify processes:

1. Constants of the signature of $MPT(A)$ is constant 0, denoting a deadlock.
2. Functions of this theory:
   1. Binary operator $+$, which denotes *choice*.
      $x + y$ behaves either as $x$ or as $y$.
   2. For each action $a \in A$ it has a unary operator $a.\_$, denoting the *action prefix*.
      If $x$ is a term, then $a.x$ first executes action $a$ and then proceeds as $x$.

These are the axioms in this theory, also called $T$-terms:

| Axioms              |      |
| ------------------- | ---- |
| $x+y = y+x$         | A1   |
| $(x+y)+z = x+(y+z)$ | A2   |
| $x + x = x$         | A3   |
| $x + 0 = x$         | A6   |

Axioms A1, A2 and A3 are often referred to as *mutativity*, *associativity* and *idempotence* respectively. Axiom A6 indicates that we avoid the choice of deadlock as long as possible. Finally, the action-prefix $a.$ binds stronger than $+$ to reduce bracket clutter.

> Example of proofs in a process theory:
>
> To proof: $a.x + (b.y + a.x) = a.x$
>
> Proof: $MPT(A) \vdash a.x + (b.y + a.x) = a.x + (a.x + b.y) = (a.x + a.x) + b.y) = a.x + b.y$

In terms of the eat marry example, it turns out that $a.(x+y) = a.x + a.y$ is not derivable from the theory $MPT(A)$. We could add it as an axiom, but since the basis of this theory forms a bisimulation relation, we don't want to do that.

### The term model

Now we have a process theory $MPT(A)$, which we can use to create a *term algebra* of $MPT(A)$. This, in turn, will be used to create the term algebra modulo bisimilarity, which is referred as the **term model**.

Algebra $\mathbb P(MPT(A)) = (\C(MPT(A)), +, (a.\_)_{a \in A}, 0)$ is called the **term algebra** for theory $MPT(A)$. 

- We use equivalence, $\equiv$, on the domain of this algebra to denote the same terms.
- $\mathbb P(MPT(A))$ is not a *model* of $MPT(A)$, as equation $MPT(A) \vdash a.0 + a.0 = a.0$ does not hold in $\mathbb P(MPT(A)):a.0 + a.0 \not \equiv a.0$.

We'll use this algebra to construct a transition system where we can perform algebraic operations on. This will be called the *algebra of transition systems* $\mathbb P(MPT(A))$.

Term deduction system for $MPT(A)$:

$$
\frac{}{a.x \overset a \rightarrow x }
\qquad \frac{x \overset a \rightarrow x'}{x+y \overset a \rightarrow x'}
\qquad \frac{x \overset a \rightarrow x'}{x+y \overset y \rightarrow x'}
$$

Note that the first deduction rule has an empty set of premises and thus is called an *axiom*. With this, we've turned the *term algebra* $\mathbb P(MPT(A))$ into a *transition-system space*. We'll always assume that the *term algebra* is accompanied by a *term deduction system* that defines a *transition-system space*.

---

**Theorem 4.3.3 - Congruence**

Bisimilarity is a congruence on the algebra of transition systems $\mathbb P(MPT(A))$.

---

> Proof
>
> According to Theorem 3.2.7, the Congruence theorem, if there is a deduction system in path format, then bisimilarity is a congruence on the induced algebra of the transition systems.

> Proof 2
>
> According to Definition 2.3.16, the bisimilarity relation needs to adhere to two requirements:
>
> 1. Bisimilarity is an equivalence relation. See Theorem 3.1.13 for a proof.
> 2. For each $n$-ary function $f$ of $\mathbb P(MPT(A))$ and for all bisimilar closed terms in $\C(MPT(A))$:
>    $p_1 \bis q_1, \ldots, p_n \bis q_n \implies f(p_1, \ldots, p_n) \bis f(q_1, \ldots, q_n)$

**Term model:** The quotient algebra $\mathbb P(MPT(A))_{^/\bis}$ is called the *term model* of $MPT(A)$.

- So all process terms that are bisimilar are grouped together, i.e. $[0]_\bis$.
- These elements are called *processes*.

Process theory $MPT(A)$ is both a sound axiomatization of the algebra $\mathbb P(MPT(A))$ and a ground-complete axiomatization of $\mathbb P(MPT(A))$.

Because of soundness:

- $MPT(A) \vdash p=q \implies p \bis q$
- $MPT(A) \vdash p=q \implies \mathbb P(MPT(A))_{^/\bis} \vDash s=t$.

Because of ground-completeness:

- $p \bis q \implies MPT(A) \vdash p=q$

### Extending current theories

Wouldn't it be great if we could extend a process theory and add constants or operators + axioms to it? These extensions may allow processes to be described that could not be described in the original theory.

Let us define $BSP(A)$, Basic Sequential Processes, as an extension of $MPT(A)$. 
$BSP(A)$ equals to $MST(A)$ and an additional constant 1. The axioms are exactly the same.

Since any closed $MPT(A)$-term is a closed $BSP(A)$-term and since there are closed $BSP(A)$-terms that are not derivably equal to closed $MPT(A)$-terms, the expressiveness of $BSP(A)$ is strictly larger than $MPT(A)$.