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

Finding the maximum independent set is NP-hard, and an algorithm exists of $O(n^2 2^n)$ to find it.

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


## Answers exam

### Question 1

**Question a:** If there is a perfect matching, then Bob wins. 

Whatever vertex Alice picks, just pick the vertex that is matched with that vertex in the perfect matching.

**Question b:** ...

There exists an unmatched vertex. Alice starts by picking the unmatched vertex, then Alice just starts picking the vertex which is unmatched and keeps picking the vertex that matches in the perfect matching.

However, if Bob wins, then there is an augmented path in the graph, thus there should exist a perfect matching.

### Question 2

#### Question A

- 2-regular
- Number of conn subgraph of size $r \leq nd^{2r}$

Pick first vertex: $n$ different possibilities. Then for the second vertex you only have $d$ choices.

Then you go back. You're making a cycle on the spanning tree. At most $2r$ edges.

Number of connected subgraphs of size $r$ $\leq $ Number of different spanning trees of size $r$ $\leq$ Number of distinct settings.

#### Question B

$\mathbb E[ \text{Number of conn subgraph of size} > \log n]$

Trick, take $r=1+\log n$ instead of $r = \log n$.

## Lecture 9 - Randomized Algorithms

**Incremental Construction:** Suppose input is $Obj_1,Obj_2, \ldots, Obj_n$. Need to computer with these objects.
For the first $i$ objects we compute a partial answer, then we include the next $Obj_{i+1}$. Sort of induction-type.

*Idea:* randomly order the objects $Obj_1, Ojb_2, \ldots, Obj_n$.

#### Problem 1 - Sorting

Given: numbers $S_1, \ldots, S_n$ which you need to sort in increasing order.
First we include $S_1$, then I put all numbers smaller than $S_1$ left of $S_1$ and the numbers higher right of $S_1$.
We do the same thing with $S_2$. No numbers can go right of $S_1$ if $S_2 \leq S_1$.

The first step takes $n-1$, since we need to calculate all numbers different from $S_1$. 
The time it takes for the second step depends on the numbers $S \in left(S_1)$. It takes $O(left(S_1))$ time.
Worst case: $(n-1) + (n-2) + \ldots + 1 = O(n^2)$. Only happens if $S_1 \leq S_2 \leq \ldots \leq S_n$ or vice versa.

However, what if we randomly permute the numbers?

Let the first $i$ numbers be already sorted.

Since everything should be equal spaced, the number of numbers between $S_{i-1}$ and $S_i$ is $\frac n {i+1}$.

$$\sum_i \frac{n-i}{i+1} \leq \sum_i \frac n {i+1}=n(1+\frac 1 2 + \frac 1 3 + \cdots + \frac 1 n) = O(n\log n)$$

Now we want to know the running time if we insert number $S_{i+1}$ given that numbers $S_1 \ldots S_i$ are already somewhere in the list, sorted.

$$\mathbb E[\text{running time when inserting $S_{i+1}$}] = \sum_{k=1}^{i+1} \frac{|I_k|}{n-i} \cdot |I_k| = O(n^2)$$, since $$\sum_k I_k = n-i$$. 

The probability that $S_{i+1}$ belongs to a specific place is $\frac{\text{Size of the interval}}{n-i}$. 
This naive method does not really work, so to speak.

When you look at a randomized algorithm, you always look at the *Expected running* time.

##### New Idea

Going back. Suppose we pick $S_{i+1}$, then the running time is the number of elements in $S_i$ and $S_{i-1}$. But what if you go backwards? Just pick any random number in $1 \leq p \leq i+1$ and we can go back in a similar running time.

What is the running time? Suppose we remove $S_{i+1}$, which is between numbers $S_i$ and $S_{i-1}$. This equals the number of elements between the interval $(S_i, S_{i+1})$ plus $|I_{K+1}| = $ number of intervals in $(S_i, S_{i-1})$.

$$\mathbb E[\text{running time}] = \sum_K \frac {I_K + I_{K+1}} {i+1} \leq 2 \sum_K \frac {I_K}{i+1} =2 \left(\frac {n-1}{i+1}\right) \approx 2 \frac n i$$

This is called **backwards analysis**.

### Problem 2 - Convex hull

Try to make the smallest set possible such that when you connect them, all the points lies in this hull.

We follow the same framework:

1. Randomly order the points.
2. Incremental construction

Assume we already inserted the first $i$ points $P_1, \ldots, P_i$. These are all points on the boundary + the points inside the convex hull $C_i$. Consider a origin point $P_o$ in the convex hull and connect them to all points outside of the convex hull. Then look at the edges where they intersect with the convex hull. These edges are kind of an obstacle.

Now we insert in $P_{i+1}$. It's possible that it lies inside or outside. How to check this?
There exists an edge between $(P_{i+1}, P_o)$. Compute the point where it intersects another edge, when considering the half-line starting in $P_o$ through $P_{i+1}$. Check if $P_{i+1}$ is within the boundary of the intersection or outside. This can be done in $O(1)$ time. 

Inside? $\rightarrow$ $C_{i+1} = C_i$. Takes $O(1)$ time.

Outside? $\rightarrow$ Consider a point $P_k$ within the $i$ points $P_1, \ldots, P_{i}$. We know that $P_k$ is part of the convex hull boundary if the hull of $(P_{k-1}, P_{i+1})$ and $(P_{i+1}, P_{k+1})$ is within the convex hull.

Quicker to look at the angle ($\leq180^o$) or check for a left turn or a right turn. It's best to first consider the points where the half-line of the origin intersected the convex hull.

What is the running time of this? This depends on the number of points we remove. Time = $O(\text{# deleted})$.
However, we also need to update the edges with the edge w.r.t. the origin. For every other point we might have to change the halfline to the origin. So in total, this takes $O(\text{# of deleted + # of updated info})$.

Total running time $$= \underbrace{\sum_{i=i}^n \text{# deleted at step $i$}}_{\leq n} + \sum_{i=1}^n \text{# updated at step $i$}$$. 

So we only need to worry about the second term. Now we want to compute at every step how many vertices we have to update. These are exactly the number of vertices of which the edge intersected the edges we removed from the convex hull. In the worst case, we have to remove $n-i$ points, as all points are behind (one of) the removed edge(s). 
Thus, $\text{# of updated points at step $i$} \leq n-1$. This results in an $O(n^2)$ algorithm, but we can go faster!

What if we randomly order the points? Going backwards, we are just going to pick a random point of the $i+1$ points (not only from the hull!).

Now, if we remove a random point and again compute the convex hull. All the points 

If we pick $P_{i+1}$, our running time is $\text{# of points whose edge intersects the edges of $P_{i+1}$}$. Suppose the convex hull has $l$ edges and $e_l$ is the number of points that intersect that edge.

The probability that we pick $P_{i+1} = \frac 1 {i+1}$. Then we have to update the number of vertices whose edge intersects the two edges of $P_{i+1}$. (If $P_{i+1}$ is on the convex hull.) This equals to $e_l + e_{l+1}$. 

$$\mathbb E [\text{# of updated at step $i$}] = \sum_{k=1}^{i+1} \frac{e_l+e_{l+1}}{i+1} \leq 2 \frac{|C_{i+1}|}{ i+1} \leq \frac{2n}{i+1} = O(n \log n)$$.

**Note:** Random sampling will *not* be on the exam.

## Lecture 10 - Random sampling - Min-cut

### Min-cut

**Min-cut:** Split the vertices in two parts $S$ and $V \setminus S$ such that the number of edges is as little as possible.

**Karger** (random sampling solution): 

#### Probability of min-cut for single specific min-cut

Idea: imagine $S$ is big, $S \setminus V$ is big and the edges between them is small. When we contract an edge, it is most likely that it is likely that it's either $(1 - \frac 2 n)(1 - \frac 2 {n-1}) \cdots (1 - \frac 2 3) = \frac 2 {n(n-1)}$ in $S$ or in $V \setminus S$.

1. Pick a random edge $e$. Contract it. (I.e. $(u, v)$ becomes a new vertex $w$.)
   This can result in more than 1 edge between vertices.
   Repeat until two vertices are left.
   Possibly more than 1 edge can be removed if two vertices have 2+ edges between them.
2. Now we have two vertices between which there are 0+ edges.
   Let $S$ be one side of the cut and $V \setminus S$ be the other side of the cut.

For $i=1, 2, \cdots, n-1$ contract a random edge $e_i$. Let $k$ be the number of edges in the min-cut en $m$ be the total number of edges. 
Note that $d(v) \geq k$ for all vertices $v$. (Because otherwise we could select $v$ as $S \setminus V$.
Also note that $\sum d(v) = 2m \Rightarrow nk \leq 2m$ 
And note that $m \geq \frac {nk} 2$

$e_1:\Pr[\text{We contract an edge not in the min cut}] = 1 - \frac k m \geq 1 - \frac{2k} {nk} = 1 - \frac 2 n$
Suppose we contract an edge in the min cut. 
​    Then we have that mincut $\geq k$.
​    Else it stays the same. 
However, we can delete more than 1 edges, so we cannot state $e_2: 1-\frac k {m-1}$

$e_2: \Pr[\text{We contract an edge not in the min cut}] = 1 - \frac{2}{n-2}$ after some proving with a $2m_1 \geq k(n-1)$.

Assume we only contracted edges from $S$ or $V \setminus S$.

$e_i : \Pr[\text{We contract an edge not in the min cut}]: 1 - \frac k {m_i} = 1 - \frac 2 {n-i+1} $

What is the probability that we have not contracted an edge in the min cut every time?

$$(1 - \frac 2 n)(1 - \frac 2 {n-1}) \cdots (1 - \frac 2 3) = \frac 2 {n(n-1)} \geq \frac 2 {n^2}$$ (after writing them down and cancelling many terms)

This is only for *one* min-cut.

#### Probability of min-cut for single specific min-cut

How many of these mincuts do we have? Most stupid upper bound is $2^n$, but because of the probability we can't have more than $\frac {n^2} 2$ mincuts. Thus, any graphs $G$ on $n$ vertices can has $\mathcal O(n^2)$ mincuts.

Each contraction might take up to $n$ timesteps because we have to update all the neighbours. There are $n-2$ contractions. Thus, running time of the algorithm is $\mathcal O(n^2)$.

Well, we just repeat this $n^2$ times where our total expected running time is $n^4$.

#### Let's make it faster!

If we can find a better way where the probability is a bit higher, then we can improve the running time of the algorithm.

So in the end, we're actually having pretty big probability terms. So the probability that we contract an edge that is not in $k$ is quite small.

Idea: start with graph $G$ of size $n$. Contract until we have $t$ vertices. 
$\Pr[S\text{ survives}] \geq (1-\frac 2 {n-1}) (1-\frac 2 n) \cdots (1-\frac 2 t) = \frac{(t-1)(t-2)} {n(n-1)} \approx \frac{t^2}{n^2}$

e^4Once we're down at $t$, we'll use a $t^4$ algorithm. 
This resue^4lts in an expected running time of $[n(n-t) + t^4] \cdot \frac{n^2}{t^2} \leq \frac{n^4}{t^2} + n^2t^2$. This is smallest for $t=\sqrt n$, then we get a running time of $n^3$.

#### Let's make it even faster!

Remember that $\mathbb P(n) = 1 - [1 - \frac {t^2}{n^2} \mathbb P(t)] ^{\frac{n^2}{t^2}}$, where the latter part is the probability of fail in one of the $\frac{n^2}{t^2}$ runs. 

What if we do this splitting again and again and again, instead of just once. 
Let us take first part at $\frac 1 2 n$. 
The probability it survives is $\frac{t^2}{n^2} = \frac 1 4$, and we do this 4 times. 

For the $\frac 2 {2^i}$, we do this $4^{i+1}$ times.

Expected running time: 

$$\sum_{i=1}^{\log n} \frac n {2^{i+1}} \cdot \frac {n}{2^i} 4^{i+1} = 2n^2 \log n$$

$\mathbb P(n) = 1-(\Pr[\text{Fail in one of 4 steps}])^4$
$\mathbb P(n) = 1-(1-\frac 1 4 \mathbb P(\frac n 2))$

$\mathbb P(n) \sim \frac 1 {\log n}$. So with at least $\frac 1 {\log n}$ probability we can get a min cut.

## Lecture 11 - Spectral Graph Theorem

### Spectral Graphs and Eigenvalues

Create a matrix that is somewhat related to the graph $G$. 
For example, an adjacency matrix, it will be 1 if there is an edge between node $u$ and $v$. Since $G$ is an undirected graph, it will be mirrored.

**Spectral graph idea**
Start with a graph $\Rightarrow$ Make a matrix about $G$ $\Rightarrow$ Forget about $G$ $\Rightarrow$ Infer information from just the matrix (Eigenvalues + Eigenvectors)

**Laplace matrix**: if there is an edge between $i$ and $j$ we put a -1, else we put a 0. Finally, the diagonal entries will be the degree of the vertex.

Any vector $x \in \mathbb R^n$, we can define $Lx \in \mathbb R^n$. When we have that $Lx = \lambda x$, we have that $x$ is the eigenvector and $\lambda$ an eigenvalue.

Let $L$ be an $n \times n$ symmetric matrix of graph $G$, then it has $n$ real eigenvalues. $\lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n$
Assume these eigenvectors to be perpendicular.

How to find these eigenvalues? In particular, the smallest eigenvalue $\lambda_1$. Consider the following expression:
$$
\lambda_1 = \min_{x \in \mathbb R^n} \frac{x^TLx}{x^Tx}
$$
The smallest $x$ will be $v_1$. So for the second eigenvalue, we want to do exactly the same thing.
$$
\lambda_2 = \min_{x \perp v_1} \frac{x^TLx}{x^Tx}
$$
The vector that will make this the smallest is $v_2$.
$$
\lambda_i = \min_{x \perp \{v_1,  v_2, \ldots, v_{i-1}\}} \frac{x^TLx}{x^Tx}
$$
If your graph is just a single edge, then $x^T L x = (x_1 - x_2)^2$.

Let $L$ be a laplace matrix between the edges, then we know that:
$$
x^TLx = \sum_{(i, j) \in E}(x_i - x_j)^2
$$

### How do we use it?

Suppose we have a graph $G$ and a laplace matrix $L$. We know that $x^TLx$ is the sum of squares. Furthermore, the smallest eigenvalue can be written as $$\lambda = \min_{x \in \mathbb R^n} \frac{x^TLx}{x^Tx}$$. Because $x^TLx$ is the sum of squares, we know that all eigenvalues must be at least 0.
$$
\begin{align*}
\lambda_1 & = 0, v_i = \left( \matrix {. \\ .\\.}\right)\\
\lambda_2 & = 0 \Longleftrightarrow G \text{ is disconnected.} \\
\cdots \\
\lambda_k & = 0 \Longleftrightarrow G \text{ has $k$ disconnected components.}
\end{align*}
$$

#### Some more theorems

$G$ is $d$-regular $\implies\lambda_n \leq 2d$

## Lecture 12 - Probabilistic Exponential time algorithms

**Stirling's approximation:** $n! \approx \sqrt{2 \pi n} (\frac n e)^n$
There, we can use that $O^*(n!) \leq O^*((\frac n e)^n)$
Also: $1+x \leq e^x$

Directed $k$-path problem: Given directed graph $G$, is there a simple path on at least $k$ vertices?
Is $G$ acyclic? $\Rightarrow$ polynomial time via DP. (Create array $A[v]$ and get the max from this DP vector. Uses the fact that $$A[v_i] = \max_{u \in N^-(v_i)}A[u]+1$$.

$$asld;​$$