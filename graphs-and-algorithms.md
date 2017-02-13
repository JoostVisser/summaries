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