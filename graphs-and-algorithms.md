# 2MMD30 - Graphs and Algorithms - Summary

[TOC]

## Lecture 1

### Administration

Grades

- Two homeworks (10% each)
  - Write down intermediate steps and thoughts.
- Midterm (30%), Final (50%)

### Some definitions

What is a graph? A graph consists of nodes, called vertices, which are connected via edges.

**Complete** graph has an edge between every vertex.

**Cycle:** Any vertex can reach itself in non-zero edges.

**Bipartite graph:** Some vertices on the left and right side. Every edge is between the left and right side.

- **Complete** bipartite graph: every possible edge between left and right.

**Connected graph:** From every vertex you can reach every other vertex.

**Tree:** Connected graph with no cycles 

Degree vertex $d(v)=$ # of edges connected to vertex $v$.

### Various different graph problems

#### Bipartite graph vertices

How do you show that a graph is bipartite?

- Basically, you have two sets, $A$ and $B$. 
- If a vertex belongs to $A$, then all it's neighbours should belong to $B$. (*2-colour*)
- For large graphs can to it proceduraly. 

How to prove a graph is not bipartite?

- Naive way, check all possible ways. $O(2^n)$

**Problem:** $G$ is bipartite iff $G$ has no odd cycles.

- Then we can just look for odd cycles.

-----

**Theorem:** $G$ is bipartite iff $G$ has no odd cycles.

-----

Three-colouring? $\longrightarrow$ NP-hard

#### Independent set

**Independent set:** Set of vertices where no two vertices are neighbours.

Given a graph with max $d(v)=p$.

The maximum independent set then has at least $I = \frac n {p+1}$ vertices. 

Finding the maximum indepentent set is NP-hard, and an algorithm exists of $O(n^2 2^n)$ to find it.

#### Vertex Cover

Subset of vertices such that all the edges have at least one of its two vertices it connects in the vertex cover. 

- I.e. there exists no edges that exist outside the set.

-----

**Theorem:** $P$ is a vertex cover iff $V-C$ is the size $I$

-----

Finding minimum vertex cover: No polynomial time exists.

#### Matching

**Matching or independent edge set:** A set of edges without a vertex in common.

- Other version: Set of edges where no edges is a neighbour of eachother.

Suppose we have a *maximal* matching of $M$, then we know that the minimum vertex cover $C$ is within $M \leq C \leq 2M$

Finding *maximal* matching? There exists a polynomial algorithm.

#### Approximating vertex cover

What if we do want to have a polynomial time algorithm? $\rightarrow$ Approximation.

Is at most $\alpha$ times the best possible minimum vertex cover. 

One idea would be to use *maximal* matching.

##### Vertex cover approx using maximal matching

Find a maximal matching and use the theorem that the minimum vertex cover $C$ is within $M \leq C \leq 2M$. Thus we have an $\alpha = 2$-approximation.

#### Relation between vertex cover and matching

We know that for the *maximum* matching $M^* \leq C$. Thus, we have that $\max M \leq \min C$. 

-----

**Theorem:** For bipartite graphs, it holds that $\max M = \min C$.

------

## Lecture 2

### Maximal matching for bipartite graph

Skipping this part since I was late, but it involves Hall's Theorem and augmented paths. Augmented paths are alternating paths with a start and end point, starting with a white edge and ending with a white edge where the current paths are green edges. 

If there still exists an augmented path, then we can flip the green and white edges thus we can increase our current matching.

We can also use the vertex cover via the following formula: $\max M = \min C$.

---

**Theorem:** if Maximal Matching $M$ is not maximum, then there exists augmented path. 
This theorem holds for both bipartite and non-bipartite graphs.

---

### Maximal matching for non-biparite graphs

We cannot use Hall's theorem for non-bipartite graphs, since we have no notion of perfect matching.

First bound for perfect matching:

$|M| \leq \frac 1 2 n$

Second bound for perfect matching after some fancy math. 

$|M| \leq \frac 1 2 (n + U - o(G \setminus u))$ where $o$ is the number of uneven groups $G$ connected to $U$.

-----

**Tutte-Berge theorem**: $\max M = \min_U \frac 1 2 (n + U - o (G \setminus u))$ 

-----

Difference with bipartite graph, we now have to deal with uneven cycles.

The proof is really difficult, but we don't need to be able to do this prove ourselves, but we do have to apply it with the running time.

## Lecture 3 & 4 - Probabilistic method

Random variable: $\mathbb E[X] = \sum_i p_i n_i$.

Independence: $\mathbb P[X=n_i \cap Y=n_j] = \mathbb P[X=n_i] P[Y=n_j]$

Linearity of Expectations: $\mathbb E[X + Y] = E[X] + E[Y]$

We use the following properties that follow:

- $\exists i : n_i \geq \mathbb E[X]$ and $\exists_j : n_j \leq \mathbb E[X] $
- Consider any event $F(X)$. 
  If this event occurs, thus we have that $\Pr[F(X)] > 0$, then there exists some value for $X$ where $F(X)$ occurs.
  Similarly, if we have that $F(X) < 1$, then there exists some value for $X$ where $F(X)$ does not occur. 





Example: $G =  (n, m)$. Delete $\leq \frac m 2$ edges such that $G$ is bipartite.

> Proof
>
> Consider biparite part $A$ and part $B$, and for each vertex $v$, we independently divide them in either $A$ or $B$ with probability $\frac 1 2$.
>
> Now we remove all edges between the vertices in the bipartites $A$ and $B$. 
>
> Let $X$ be number of 

- Make a random variable $X_e = 1\{\text{$e$ is deleted}\}$
- $X = \sum_e X_e \leq \frac 1 2 m$
- $E[X] = \sum_e \mathbb E[X_e] = \sum_e 1\cdot \mathbb P[X_e=1]  = \sum_e \frac 1 2= \frac 1 2 m$
- Thus there $\exists_m :  m \leq \mathbb E[X] = \frac 1 2 m$.

### Ramsey graphs

Suppose we have 6 people, that are either friends with each other or enemies from each other.

- $\exists a, b, c :$ all are friends or all are enemies.

> Proof
>
> We model the problem as a graph $G=(V,E)$ such that each person is a vertex and an edge of them is red if they are enemies from each other and green if they are friends from each other. Notice that this will be a complete graph $K_6$.
>
> Consider a vertex $v$ and look at the edges. Consider the edges where the colour are in the majority for this vertex, thus there will be at least 3 edges of this colour. Suppose $v_1$ connects to $v_2$, $v_3$ and $v_4$ with this colour, then we know that the edge $ (v_2, v_3)$, $(v_3, v_4)$ and $(v_2, v_4)$ cannot be of the same colour, otherwise you would get a triangle of vertices $v_1, v_x$ and $v_y$. However, now we have a triangle of the other colour of $v_2$, $v_3$ and $v_4$, thus we get a contradiction hence it isn't possible.

If we generalize to $n$ people with $p$ of the same colour, then we define the ramsey number:

$R(p, p) = \min_n$ such that $K_n$ contains red $K_p$ or green $K_p$.

Now we want to show that $R(p, p) > 2^\frac p 2$. We just have to show that $\exists p:R(p, p) = 2^\frac p 2$ to show this.

> Proof
>
> We want to show that $\mathbb E[\text{# of monochromatic $K_p$} ] < 1$.

Another Ramsey's number:

$R(p, s) = \min_n $ such that $K_n$ contains red $K_p$ of green $K_s$

### Independent sets

$G$ of maximum degree $d$, then $G$ has an independent set of size $\geq \frac n {d+1}$

> Remember, an **Independent set** is a set of vertices where no two vertices are neighbours.

Now, let's see s slightly better bound: $|IS|\geq \sum_i \frac 1 {d_i + 1} \geq \frac n {d+1}$

Define a random order of vertices. Let's say $v_1 > v_2 > v_3 > \ldots > v_n$.
I pick the first vertex. I remove all the vertices which are attached to it and proceed to the next vertex I have.

Let $X_i = 1\{ \text{If $i$ is in I.E.} \}$

And let $X = \sum_i X_i$.

$E[X_i] = \Pr[i \in \text{I.S.}] \geq \Pr[\text{i has no neighbour in $v_1, \ldots, v_{i-1}$}]$

​            $ = \Pr[\text{$i$ is the first vertex among ($i, a_1, a_2, \ldots, a_{d_i}$)}] = \frac 1 {d_i + 1}$.

### Probabilistic method to algorithm

This is not always possible, but sometimes it's possible.

Show a greedy step that always performs $\geq$ then the probabilistic step, thus you have a method that performs better. Example: at the delete the $\frac m 2$ algorithm we can create an algorithm where we look at vertex $v$, if the majority of the edges of $v$ go to $A$, then we will put the edge in $B$ so we will remove less than half edges, else we put in $B$ and remove less than half the edges.

### Dominating sets

#### Method 1

**Dominating set:** Each vertex in $V \setminus S$ has an edge to some vertex in $S$.

Show that, for a graph $G$ with $\min $ degree $d$: $\exists a : |DS| \leq (c \frac{\log d} d)n = x$

Suppose we have a vertex $v$, then we choose with probability $p$ that it is in $S$ and $1-p$ that it is in $V \setminus S$. 

We want to show that $\mathbb E[|DS|]  \leq (c \frac{\log d} d)n = x$.

We want to calculate $\Pr[|S| \leq x \cap \text{S is a DS}]$ > 0. 

If we prove that $\Pr[|S| \leq x] > \frac 1 2$ and $\Pr[\text{S is not DS}] < \frac 1 2$

Because one event happens with $>0.5$ and another event happens is $< 0.5$, then there must be some part where the first holds and the second does not hold. 

Let $X_v = 1\{ \text{$v$ and all the neighbourhood of $v$ is outside of $S$}\}$

Since a vertex has a chance of $1-p$ to be outside of $S$, so the expected value $\mathbb E[X_v] = n(1-p)^{d+1}$.

We want to have that $\Pr[X\geq 1] < \frac 1 2$, so we have to use Markov's:

We have to use Markov's: $\Pr[X > a] \leq \frac {E[X]} a$

After some math, this results in the bound $\frac{n \log n}{d+1}$

#### Method 2 - Probabilistic Method with Alterations 

**Step 1:** Include each set in $S$ with probability $p$, and in $V \setminus S$ with $(1-p)$
Let $T$ be all vertex that cause $S$ to not be a dominating set. 

**Claim:** $S \cup T$ is a DS.

$\mathbb E[|S \cup T|] = \mathbb E[|S|] + E[|T|] = np + E[|T|]  \leq np +  n(1-p)^{d+1}$

We want to find the lowest expectation, so we want to minimize p, i.e. differentiate: $\frac \partial {\partial p} np +  n(1-p)^{d+1} = 0$

After some math, we get that this results in $\leq \frac{n \log d} d$.

So sometimes it helps to not divide everything random but look at the problem in a different way.

### Lovasz Local Lemma

$A_1, A_2, \ldots, A_k \rightarrow$ bad events, independent and let it happen with probability $\Pr[A_i] = P$

We want none of these to occur, i.e. $\Pr[\bigcap_k \bar A_k] > 0$ ($\bar A$ is that $A$ does not hold.)

Then, let us define $X_i = 1${If $A_i$ occurs} and $X = \sum_i X_i$

Now, we want to show that $\mathbb E[X] < 1$, where we can calculate $\mathbb \sum_iE[X_i]$ = kp < 1, so $p < \frac 1 k$.

Then it holds that $\Pr[\bigcap_k \bar A_k] = (1-p)^k > 0$.



Now, what if these variables $A_1$ are almost independent, so the event only depends on a small number of events. 
We can actually draw an independent graph where an edge determines non-independence and no edge determines independence. If this graph is e.g. $4$, then an event is only determined by at most 4 other events. 
Let this graph be of a degree of at most $d$.

**Lovasz Local Lemma:** $epd < 1 \Rightarrow \Pr[\bigcap_k \bar A_k] > 0$

### This problem

Suppose we have vertex $s$ and $t$ and there are $m$ possible paths from $s$ to $t$.
Furthermore, you know that each path $P_{1 \leq i \leq m}$ intersects $r$ other paths.

We want to sent $n$ packets from $s$ to $t$ such that no intersection of the paths of the packet happen. What is the maximal number of packets we can send?

For each $i$, choose a random path. $X_{ij} = 1 \{\text{if } P_i \cup P_j\}$, so 1 if $i$ and $j$ have an edge in common thus it cannot hold.

$\sum_{i,j} \mathbb E [X_{ij}] < 1$, where $\mathbb E[X_{ij}] = \frac{r+1} m$, resulting in the end to $n < \sqrt{\frac m r}$

What if we want to solve it with the Lovasz Local Lemma?

We'll take a similar $X_{ij}$, giving rise to a similar porbability of $p = \Pr[X_{ij} = 1] \leq \frac{r+1} m$

We want to know when $epd < 1$, so what is $d$? In total, $X_{ij}$ depends on at most $2n-1$ events, so $d=2n$.
(Why? Because the fact that the routes of packets 1 and 2 $X_{12}$ has an edge in common does not depend on 

$\Rightarrow$ $n \leq \frac{1}{2e} \cdot \frac m r$

### K-Sat

$(x_1 \vee x_2) \wedge (x_5 \vee \bar x_6) \wedge \cdots \wedge  (\bar x_n \vee x_8)$ is satisfiable?

2-Sat is solvable in polynomial time, whereas K-Sat isn't.

Assume: 

1. Every clause has $k$ variables.
2. Each variable occurs in $\leq \frac{2^k} {100k}$ clauses. So each variable does.

Say we have $m$ clauses, for every clause we make a random variable $Y_i = 1\{\text{if clause $i$ is not satisfied}\}$.
A clause is not satisfiable if all are 0. There is only 1 possible assignment in which a clause cannot be satisfied, which results in a probability of $p = \Pr[Y_i = 1] = \frac 1 {2k}$.

Two clauses depend on eachother if they share at least one varable. In total, there will be at most $k \cdot \frac{2^k}{100k}$ dependencies, so $d = \frac {2^k} {100}$.

An algorithm to find this is to:

1. Randomly assign all variables. 
2. For each clause which is false, pick another random assignment.
3. If $epd < 1$ works, then the algorithm works. 
   In about $n$ steps, it should terminate, as the number of new clauses that can be unsatisfied is bounded.

