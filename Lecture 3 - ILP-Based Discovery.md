## Lecture 3 - ILP-Based Discovery

### Motivation

Why pick AlphaMiner or Inductive miner or even the new ILP Miner?

- AlphaMiner cannot handle infinite-transitions which can happen at any time. Also, improper completion. 
- The inductive miner wrongly notices that the *send reminder* can happen before send invoice. We can also skip the payment but still prepare for delivery. (Problem is because of the dependencies which cannot be handled over a long term.)
- The ILP Miner actually finds the *exact* model!

### Language Based Regions

If we ignore how many times a sequence occurs, then we just get a set of sequences. A set of sequences is called a *language* $\mathcal L$.

**Region Theory:** Can find *complex* control-flow structures.

- By adding places that **constrain behaviour**, yet allow for **observed behaviour**.
  - Contrain behaviour: limit the behaviour of various transitions.
  - Observed Behaviour: It should never violate any of the given processes.
- Problem: To actually find out which places are nice to add and which are not.

Region = place $p$.

Given a place $p$, and any trace $t$ in the log. Then $p$ can never contain a negative amount of tokens at any point in time when 

Prefix-closure of a set of sequences:

- Contains *all possible prefixes* of any of the sequences.
- Example: $\langle a, b, c, d \rangle$ then $\langle \rangle , \langle a \rangle, \langle a, b \rangle, \langle a, b, c \rangle \in \mathcal L$

What is a region?

- The $x$ vector corresponds to incoming arcs and the $y$ vector corresponds to the outgoing arc. In the $x$ vector, if it's 1 then an incoming arc goes from a place to $c$, then 0 otherwise. Same for $y$ and outgoing arcs.


- Final boolean $c$ is the initial token. 
- Thus, a region is just a mathematical definition of a place.
- $w(t) = 1$ if it occurs in the word $w$, and $w'(t) = 1$ if it occurs in the prefix of the word, i.e. in $w'$.

Finding (Relaxed-Sound) WF-Nets

We can actually use an alternative definition with vectors and matrices! This is computationally much faster.

- $y_a= 1$ if place $c$ has an outgoing edge to activity $1$.

ILP = Integer Linear Programming!

We know given an event log$\implies$Linear equations.

We still need to define what we are looking for...

#### Example

$L = [\langle a, b, d \rangle ^5, \langle a, c, d \rangle ^3]$
$\mathcal L = \{ \langle a, b, d \rangle, \langle a, c, d \rangle \}$
$\mathcal {\bar L} = \{\langle a, b, d \rangle,  \langle a, c, d \rangle, \langle a, b \rangle, \langle a, c \rangle, \langle a \rangle , \epsilon \}$
$T = \{ a, b, c, d \}$ (All transitions, generally called **activity**)

### Finding (Relaxed-Sound) WF-nets

We want some sort of cost function $J$ that needs to be minimized as long as it satisfies some certain constraints. 

>  *Note:* removing output arcs and adding input arcs cannot jeopardize the feasibility of a region/place.

We want to find regions that are minimal in terms of incoming and outgoing arcs. We want to find regions that:

- Are not redundant.
- Are minimal, i.e. we'd rather have two separate minimal places than a merged place.
- Minimizes the number of time-steps that a token will reside for *each place*.

From the prefix closed language, you can actually deduce each element!

$c$ is the size of the prefix closure.
$a$ is the number of times $a$ is in the prefix closure.
Similar for $b$.

