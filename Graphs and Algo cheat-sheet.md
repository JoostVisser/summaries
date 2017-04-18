# Graphs & Algo cheat-sheet

**Check bipartite:** Look for odd cycles. $G$ is bipartite iff $G$ has no odd cycles. (2-colour, $P$)
**Independent Set:** $I \subseteq V$is independent iff $\neg \exists$ edge between vertices in $I$. $NP$

- Max degree $d$, then $\max$ independent set $|I| \geq \frac n {d+1}$. 

**Vertex Cover:** Set $S \subseteq V$ such that every edge has one vertex in $S$. $NP$

- $S$ is a vertex cover iff $V \setminus S$ is independent.

**Matching:** Set $M\subseteq E$ such that no vertex $V$ has more than one edge in $E$. $P$
(Also called *independent edge set*.)
**Perfect Matching:** Every vertex is exactly incident to one edge in $M$.
$X$**-saturating matching**: Match all vertices of $X$ in a bipartite graph $(X, Y)$.

- $\max $ Matching $\leq$ $\min $ Vertex Cover. (Equality holds for bipartite graphs.)
- Maximal matching of $M$? Then $\min $ vertex cover $|C|$ is: $|M| \leq |C| \leq |2M|$ 
- **Hall's Theorem**: $G$ has an $X$-saturating matching iff $\forall S \subseteq X$: $|N(S)| \geq |S|$
- **Tutte-Berge theorem:** $\max M = \min_U \frac 1 2 (n + U - o(G \setminus U))$
- **Tutte's matching theorem:** $G$ has a perfect matching iff  $\forall U\subseteq V$: $o(G \setminus U) \leq |U|$ 

**Dominating set:** Set $S \subseteq V$ such that each vertex is either in $S$ or has a neighbour in $S$.

### Probability theory

Useful inequalities:

- $\binom n k \leq \frac{n^k}{k!}$ and $\binom n k \leq (\frac{en} k)^k$ and $\binom n k \leq 2^n$ and finally for $p \in [0, 1]: (1-p)^n \leq e^{-pn}$.
- $n! \approx \binom n e$ for $O^*(n)$ algorithms. 

**Markov's inequality:** $\Pr[X \geq a] \leq \frac{\mathbb E[X]} a$ ($X$ non-negative and $a>0$.) 

**Lovasz Local Lemma (LLL):** Consider $n$ bad events $A_1, \ldots, A_n$. 

- It also holds that $\forall_{1 \leq i \leq n} : \Pr[A_i] \leq p$ and each $A_i$ depends on at most $d$ other events $A_j$.
- Then $ep(d+1) \leq 1 \implies \Pr[\bigcap_i \bar A_i] > 0$.

Probabilistic idea is to randomly assign (or assign with probability $p​$) stuff:

1. $\exists i : n_i \geq \mathbb E[X]$ and $\exists_j : n_j \leq \mathbb E[X] $. This can be useful for expected sizes of certain sets.
2. Consider event $A(X)$. If $\Pr[A(X)] > 0$, then there must exist a value of $X$ where $A$ occurs.
   Similar vice-versa.

### Algorithms

