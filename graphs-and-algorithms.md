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

## Lecture 3 - Probabilistic method

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

**Dominating set:** Each vertex in $V \setminus S$ has an edge to some vertex in $S$.

Show that, for a graph $G$ with $\min $ degree $d$: $\exists a : |DS| \leq (c \frac{\log d} d)n = x$

Suppose we have a vertex $v$, then we choose with probability $p$ that it is in $S$ and $1-p$ that it is in $V \setminus S$. 

We want to show that $\mathbb E[|DS|]  \leq (c \frac{\log d} d)n = x$.

We want to calculate $\Pr[|S| \leq x \cap \text{S is a DS}]$ > 0. 

If we prove that $\Pr[|S| \leq x] > \frac 1 2$ and $\Pr[\text{S is not DS}] < \frac 1 2$

Let $X_v = 1$

We have to use Markov's: $\Pr[X > a] \leq \frac {E[X]} a$