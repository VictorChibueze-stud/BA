**1. What is Phylogenetics?**

*   **Goal:** Phylogenetics is all about figuring out how different species are related to each other based on their evolutionary history. It's like building a family tree, but for species.
*   **How it Works:** We compare features of the species, looking for similarities and differences. These features can be physical things or genetic data.
*   **Assumption:** The basic idea is: if two species are very similar, it likely means they share a more recent ancestor, and therefore they are genetically close.
*   **Phylogeny:** The term "phylogeny" just refers to these evolutionary relationships.
*   **Phylogenetic Tree:**  We usually display these relationships in a diagram called a phylogenetic tree. This tree helps us visualize how different organisms are connected over time.
*   **Classic Phylogeny:** This traditional approach used physical features, like shape, size, color, number of legs, etc.
*   **Modern Phylogeny:** Today, we rely on genetic data, primarily from DNA and protein sequences. This gives a more direct and accurate view of evolutionary relationships, because we can look at the source code for life.
*  **Features:** Modern phylogenetics usually compares the DNA or protein sequences. The positions inside the sequence are called “characters” or “features”. They are found in conserved regions of multiple sequence alignments.

**2. The Tree of Life**

*   **What it is:** The tree of life aims to visualize the complete evolutionary history of all living things.
*   **LUCA:**  The acronym LUCA stands for "Last Universal Common Ancestor." This is the root of the tree of life. It is the most recent ancestor that all known living things share.
*  **Branches of the Tree:** The tree is divided into three main branches: Bacteria, Archaea, and Eukaryota. All of these branches come from LUCA. These branches then divide into more species.

**3. Approaches to Phylogenetic Tree Construction**

There are three main ways to build phylogenetic trees:

*   **Distance-Based Methods:**
    *   **Key Idea:** These methods rely on a measure of "distance" between species. The goal is to build a tree that best reflects the calculated distances between all the species.
    *   **Example:** You might use the number of differences in their DNA sequences as a distance measure.
*   **Character-Based Methods:**
    *   **Key Idea:** These methods focus on analyzing individual characteristics or traits (characters) of each species.
    *   **Example:** A trait could be whether an animal is a mammal or not, or whether a cell is unicellular or multicellular. Species are grouped together based on how similar their characteristics are.
*   **Probabilistic Methods:**
    *   **Key Idea:** Instead of relying on distances or characters, these methods use a probability model. They aim to find the tree that has the highest likelihood (probability) of producing the observed data.
    *   **Bayesian approach:** In this approach the likelihood is combined with prior beliefs about what trees are more likely, resulting in a posterior probability.

**4. Phylogenetic Trees**

*   **Basic Structure:**
    *   **Nodes:** Each node in the tree represents a species (either real or a hypothetical ancestor).
    *   **Edges (or Branches):** Edges represent genetic connections and the relationships between species.
*   **Key Concepts:**
    *   **Leaf Nodes:** These are the actual, real-life species. They are located at the ends of the branches.
    *   **Internal Nodes:** These represent hypothetical common ancestors of groups of species. They are located at branch junctions inside the tree.
*   **Tree Types:**
    *   **Rooted Tree:** Has a root node representing the most ancient ancestor, defining the direction of evolutionary time.
    *   **Unrooted Tree:** No root, indicating that the direction of evolution is unknown. Just shows the relationships between the sequences.
    *   **Binary Tree:** Each node has at most two children/subnodes (except the leaves, which have none).

**5. A Simple Solution?**

*   **Trivial approach:** The most straightforward way to solve phylogenetic inference is to calculate the score of all possible trees, and choose the tree with the highest score.
*   **Problem:** The number of possible trees grows extremely rapidly with the number of species. This is because the number of trees increases with the “double factorial” of the number of species (for binary trees), `(2n - 3)!!`.
*   **Super-Exponential Growth:** Even with only 20 species, there can be around 10²¹ possible rooted binary trees. This makes it impossible to try all trees for any reasonably-sized dataset.
*   **NP-Complete:** Because finding the best tree is so difficult, it is an NP-Complete problem. In practice it means that it can't be solved quickly with algorithms.

**6. Number of Nodes and Edges**

*   **Rooted Trees:**
    *   **Inner Nodes:** If there are `n` leaves (species), there will be `n - 1` internal nodes.
    *   **Total Nodes:**  Thus, the total number of nodes is `(n - 1) + n = 2n - 1`.
    *   **Edges:** The number of edges will always be `2n - 2` edges (not counting the edge above the root).
*   **Unrooted Trees:**
    *   **Total Nodes:** `2n - 2` nodes.
    *   **Edges:** `2n - 3` edges.

**7. Distance-Based Methods**

*   **Distance:** These methods start by defining a "distance" between all pairs of species.
*   **Goal:** The goal is to build a tree that explains these observed distances, and to have the distances in the tree as close as possible to the measured ones.
*   **Simplification:**  Distance-based methods lose some information because they reduce all the information about the sequences to a single number of distance.
*   **Assumption:** Most of the information is actually contained in the pairwise distances between the sequences.

**8. Least Squares Methods**

*   **Idea:** They try to approximate the observed distances between species with the distances in the tree.
*  **Goal:** To find a tree `T` such that its leaves correspond to the `n` species, that has distances `dᵀᵢⱼ` as close as possible to the observed distances between species `dᵢⱼ`. This is done by minimizing the Sum of Squared differences, `SSQ(T)`.
*   **SSQ(T):** It's a measure of the discrepancy between the observed distances (measured from the real sequences) and the tree distances (distances between leaves on the tree).
*   **Difficult Problem:** Finding the best tree (the one with the lowest SSQ) is a very difficult problem (NP-complete) because it is a discrete problem of finding the best tree topology.
*   **Heuristics:** There are two main efficient approximation algorithms:
    *   **UPGMA:**  A simple approach but works under strict conditions.
    *   **Neighbor-Joining:** A faster method that tends to work better than UPGMA.
*   **Ultrametric Trees:**
    *   A special type of tree where all the leaves have the same total distance to the root.
    *   Assumes that all species evolved at the same rate, which is unlikely in real life.
   *    In this special case UPGMA is guaranteed to find the correct topology.

**9. The Least Squares Tree Problem**

*   **Formal Definition:**
    *   **Input:** The observed distances `dᵢⱼ` between all pairs of species arranged in a matrix D.
    *   **Question:** Build a tree with leaves that represent all species, such that minimizes `SSQ(T)`.
*   **NP-Complete:** As mentioned before this optimization problem is NP-complete, so it can't be solved easily.

**10. Efficiently Solvable Special Cases**

*   **Additive Distance Matrices:** These matrices satisfy a very particular property: It is always possible to build a tree such that the distances between the leaves on the tree are equal to the observed distances.
*   **Additive Property:** In additive trees, the distances are just a sum of edge lengths.
*   **Neighbor-Joining:** When the distance matrix is additive, the neighbor joining algorithm is guaranteed to find the correct topology.
*   **Key Point:** In real life, it is very difficult to have a perfect additive distance matrix, and we have to resort to heuristics.

**11. UPGMA**

*   **What it is:**  UPGMA (Unweighted Pair Group Method with Arithmetic Mean) is a simple and fast method for creating phylogenetic trees. It is a heuristic, so it does not guarantee finding the best tree.
*   **How it Works:**
    *   It starts with all the species in their own cluster.
    *   It iteratively joins the two nearest clusters and merges them into a new cluster.
    *   The distances to this new cluster are calculated as a weighted average of the distances from the two merged clusters.
    *   The algorithm stops when there is just one cluster remaining.
*   **Assumption:** UPGMA assumes that all species evolved at the same rate (which is not always true).
*   **UPGMA and Ultrametric Trees** UPGMA can only generate a ultrametric tree.

**12. UPGMA: Analysis**

*   **Metric:** A metric distance function follows three rules:
    *   `d(x,y) > 0` if x and y are different and is 0 if they are equal.
    *    `d(x,y) = d(y,x)` the distance is symmetric.
    *   Triangle inequality: `d(x,y) <= d(x,z) + d(z,y)`.
*   **Ultrametric:** It has an additional, more strict restriction:
    *   `d(x, y) ≤ max (d(x, z), d(y, z))`.
*   **Clocklike Trees:** Ultrametric trees assume a constant rate of evolution, similar to a molecular clock.
*  **Key Point:** If the distances are ultrametric UPGMA is guaranteed to find the correct topology.

**13. Additive Trees**

*   **Ultrametric Trees and Reality:** Ultrametric trees are a simplification of reality because mutations do not always happen at the same rate.
*   **Additive Trees:** Additive trees are a generalization of ultrametric trees, where the number of mutations is proportional to the time of evolution (or genetic distance). Additive trees do not have the constraint of equal evolutionary times for all the species.
*   **Generalization:** This type of tree relaxes the assumption of molecular clock, making the trees more accurate for many types of data.
*   **Unrooted:** In this framework the trees are usually unrooted, because there is no information about the root.
*   **Binary trees:** the internal nodes (except leaves) have a degree 3, meaning that it is a binary tree.

**14. Additive Distance Matrix**

*   **Definition:** A distance matrix is additive if there is a tree that exactly matches all the given distances. So if you add the branch lengths you get the distance in the matrix.
*  **SSQ(T) :** The sum of squares is zero `SSQ(T) = 0`.
*   **Relationship:** All ultrametric matrices are additive, but the reverse is not true.
*   **Four-point condition:**  A way to test if a matrix is additive, by using four arbitrary species.  Two of the summed distances will have the same value and it is larger than the third one.
*   **Triangle Inequality:** The four-point condition is a generalization of the triangle inequality.

**15. Neighbor-Joining**

*   **What it is:** An approximation algorithm for building phylogenetic trees. It tries to approximate the least squares tree, by assuming additivity.
*   **Key Difference from UPGMA:** Does not assume a molecular clock, so it is more accurate for more realistic trees.
*   **How it Works:**
    *   It looks for two species that are directly related and groups them together.
    *   Then, it iterates by grouping the clusters until one final cluster remains.
*   **Distance Computation:** When grouping two species into a cluster the new distances to other species needs to be computed, `dkm`.

**16. Correcting Distances**

*   **Problem:** Just picking the two closest leaves is not accurate.
*   **Solution:** Before picking the two closest leaves you compute "corrected" distances. This correction takes into account not just that the leaves are close together, but also that they are far away from other species.
*   **How It Works:**
    *   Compute the average distance of every node `i` to all other nodes, called `uᵢ`.
    *   "Correct" each distance `dᵢⱼ` by calculating `qᵢⱼ = dᵢⱼ - (uᵢ + uⱼ)`.

**17. Neighbor Joining Theorem**

*   **Theorem:**  If the original distances are additive, then the two species with the minimal corrected distance (`qᵢⱼ`) will be neighbors on the true tree.
*   **Key Point:** This theorem justifies why neighbor-joining can find the true tree under the right conditions (additive matrices).
*   **Branch Length Computation:** Branch lengths for a newly created node *k* are calculated using `dkm`.

**18. Neighbor-Joining: Distance Computation**

*   **Problem:** Finding the distance from a new node to the other nodes is done in the neighbor-joining algorithm using the formula `dᵢₖ = (dᵢⱼ + dᵢₘ - dⱼₘ) / 2`. The problem is that the value of `dᵢₖ` depends on what value you chose for `m`.
*  **Solution:** Instead of choosing a random `m` we can average over all the values of `m` that are different than `i` and `j`. By simplifying you can arrive at the expression: `dᵢₖ = (dᵢⱼ + uᵢ - uⱼ) / 2`

**19. Reconstructing Trees from Non-Additive Matrices**

*   **Realistic Data:** In the real world, we don't always have perfect additive distance matrices.
*   **Can We Still Use NJ:**  You can still use Neighbor-Joining even if the matrix is not perfectly additive.
*   **Issue:** There is no guarantee that the output is accurate. The tree topology might be wrong, and even the same input with different ways of resolving ties could lead to different trees.

**20. Almost Additive Distance Matrices**

*   **Definition:** A matrix is “almost additive” if there exists an additive matrix `D` such that the distance between the matrices `D` and `D’` is small. This distance is computed as the max distance between all entries in the matrices, where the max distance is less than a quarter of the smallest edge in the real tree.
*   **Theorem:** If `D’` is “almost additive” then Neighbor Joining will create a tree `T’` with the same topology as the real tree `T`.
*   **Key Point:** Neighbor Joining can find the correct tree even when there are some errors in the distance matrix, as long as it is “almost additive.”

**21. Character-Based Methods**

*   **Focus:** Uses characteristics (characters or traits) of species to make the tree.
*   **Input:**
    *   `n` species
    *   `m` characters for every species.
    *   Each character has a discrete value that belongs to the set of possible values `Σ`
*   **Goal:** Find a tree that explains the observed characteristics and their distribution across species.
*   **Assumptions (Simplifications):**
    *   Characters evolve independently of each other.
    *   After two species diverge, they evolve independently.

**22. Character-Based Methods: Parsimony**

*   **What it is:** A scoring function that penalizes the number of changes that must have occurred in evolution.
*  **Parsimony Score:** The parsimony score is the number of times that the value of the characters have changed along the edges of a tree. The best tree is the one with the smallest score, i.e., fewer evolutionary changes.
*   **Goal:** To find the tree that requires the fewest evolutionary changes to explain the observed data.
*   **Formal Definition** The parsimony score is computed as `S(T) = ∑|{j : vj != uj}|` where the sum is done over all the edges of the tree, and where j is an index over characters.

**23. Weighted Small Parsimony**

*   **Improvement:** Weighted parsimony is an extension of parsimony that accounts for the fact that all changes are not equal.
*   **Costs:** Each change has an associated cost, that is characterized by `Cᵉᵢⱼ`, where `e` is the character, and `i` is the state before the change, and `j` is the state after the change.
*   **Goal:** Minimize the total cost of the tree considering the weighted costs, given the tree structure and the leaf values.
*   **Problem:**
    *   Find the minimum cost of a tree with a certain topology.
    *   Find the labels for the internal nodes.

**24. Recall: Tree Traversals**

*   **Tree Traversal:** Systematic ways of visiting all nodes in a tree.
*   **Depth-First Traversal:**
    *   **Preorder:** Visit node, then left sub-tree, then right sub-tree.
    *   **Inorder:** Visit left sub-tree, then node, then right sub-tree.
    *   **Postorder:** Visit left sub-tree, then right sub-tree, then the node itself.

**25. Sankoff's Algorithm**

*   **What it is:** A dynamic programming algorithm to solve the weighted parsimony problem efficiently.
*   **Step 1 (Postorder):**
    *   Compute the minimum cost of the subtree under each node `v`, considering that the value of the node is `t`. This is computed recursively by doing a postorder traversal.
    *   The minimum cost of the subtree below the node *v* is stored as `Sᵗc(v)`
*   **Step 2 (Preorder):**
    *   Determine the optimal value for each internal node. You start at the root and work your way down in preorder.
    *   The value of a node *v* is determined as `v_c = argmin_t(C^u_ct + S^t_c(v))`, where *u* is the parent of *v*.

**26. Large Parsimony**

*   **Final Goal:** The problem of finding the best tree, not just finding labels for a given tree (Weighted small parsimony problem).
*   **Input:**  A set of species, characterized by characters in a matrix M
*  **Question:**  What is the best tree for these species, considering the parsimony score.
*   **Complexity:** This problem is NP-Hard, so it can't be solved efficiently for a large amount of species.
*   **Remark:** It can be weighted or non-weighted, but the difference is not essential.
*   **Heuristics:** There are several approximation methods that can find a solution fast in practice.

**27. Branch and Bound**

*   **Optimization Strategy:** An algorithm that searches for the best solution by exploring and eliminating parts of the search space.
*   **Search Space:** The search space is structured as a tree. The leaves of the search tree represent the possible trees you are searching for.
*   **Search Tree:** Nodes are usually partial solutions to the problem.
*   **Monotonicity:** Requires that the search space be monotonous, meaning that if the score of the partial tree is already high, all the subtrees will also have scores higher than the current node.
*   **Optimal Solution:** It is guaranteed to find the optimal solution, but it can take a long time in the worst-case scenario.
*  **Pruning:** During the algorithm you keep a bound (C'). If you encounter a node with a score higher than this, you prune the node, so you don't have to consider this entire part of the tree.

**28. Branch and Bound for Parsimony**

*   **Search Space:** The search space here is all possible trees, and the branch and bound explores this search space.
*   **Tree Level:**  The level of the search tree `k` represents all possible phylogenetic trees with `k` leaves for the first `k` species.
*   **Child Nodes:** The child nodes are generated by all phylogenetic trees constructed by adding the next species to the existing trees.
*   **Monotonicity:** As you add more species to the tree, the parsimony score can never decrease, so the search space is monotonous.
*  **Improvement:** Although Branch and Bound does not change the worst-case complexity, in practice it usually speeds up the search significantly.
*  **Practical Strategy:** It is recommended to start with a fast algorithm like Neighbor-Joining to get an initial topology, and use the resulting score of this tree as a first bound.

**29. Maximum Likelihood Methods**

*   **What it is:** A type of probabilistic method, where the tree is evaluated based on how well it describes the data.
*   **Likelihood Function:** `P(Data|Parametrized model)`. It represents the probability of seeing the data given a specific tree and the set of parameters. It is a function of the parameters of the model, which are the tree structure and the edge lengths.
*   **Parameters:** In phylogenetic inference the parameters include the tree's topology and its branch lengths.
*   **Problem 1 (Likelihood):** To compute the likelihood of a given tree, considering the data.
*   **Problem 2 (Maximum Likelihood Inference):**  To find the tree with the parameters that gives the maximum likelihood of the data (most likely).

**30. Computing the Likelihood of a Tree**

*   **Labels:** Sets of character values for each species and for the internal nodes.
*   **Reconstruction:** The entire tree, including the labels of internal nodes.
*   **Branch Length:** The length of an edge represents the biological time or genetic distance between two species.
*   **Assumptions:**
    *   Each character evolves independently from the others.
    *  The evolution of labels is a Markov process, where the probability of a new state depends only on the parent node and the branch lengths.
    *  Character frequencies are fixed over time.

**31. The Maximum Likelihood Problem**

*   **Input:** The character values for all species, and the topology of the tree and its branch lengths.
*  **Question:** Given a tree, find the likelihood of the tree given the input data, `L= P(M|T,t)`, and assuming that all characters are independent.

**32. Likelihood of a Tree**

*   **Complexity:** To find the likelihood of a tree, you need to calculate all possible labelings for the internal nodes.
*   **Formula:** The likelihood of the tree is a sum over all possible reconstructions of the product of the probabilities of going from a parent to a child nodes.

**33. Computing the Likelihood**

*   **Dynamic Programming:** A dynamic programming algorithm to compute the likelihood of the tree.
*   **Subtree Likelihood:**  `Cj(x,v)` represents the likelihood of the subtree with root *v*, given that at the node *v* the character `j` has the value `x`.
*   **Subtree Calculation:**  The algorithm will do a post order traversal of the tree, and will calculate recursively the value of `Cj(x,v)` using `Cj(y,u)`.

**34. Maximizing the Likelihood**

*   **Branch Lengths:** Given a tree topology, find the branch lengths that maximize the likelihood. There is no analytical solution for this, so you have to rely on numerical methods such as conjugate gradients.
*   **Tree Topology:** Finding the tree topology that maximizes the likelihood is even harder.
*  **EM-like methods:** Iterative algorithms that simultaneously optimize topology and branch lengths.

**35. Bayesian Approaches**

*   **Beyond Maximum Likelihood:** Instead of just finding the best tree, Bayesian methods try to understand the *distribution* of trees, by computing the posterior probability.
*   **Posterior Distribution:** It is the probability of each tree given your observations and your prior beliefs.
*   **Formula:** `P(T,t|M) = P(M|T,t) * P(T,t) / P(M)`.  It is calculated by combining the Likelihood with the prior.
*   **Sampling:** It's usually impossible to compute the distribution analytically, so Bayesian methods will try to sample from the distribution.
*   **Law of Large Numbers:** If you take enough samples, the frequency of a property in the samples is a good estimate of the posterior probability of that property.

**36. The Metropolis Method**

*   **What it is:** A way to draw samples from the posterior distribution.
*   **Proposal Distribution:**  It uses a procedure `f`, which generates a new tree using the current tree.
*  **Algorithm:**
    1.  Start with a tree (T, t).
    2.  Generate a new tree (T', t') using `f`.
    3.  Accept the new tree if it is better, otherwise, accept with a probability depending on how bad it is.
    4. Repeat until enough samples are taken.

**37. A Proposal Distribution for Trees**

*   **Traversal Profile:**  A profile where the tree is traversed in order, allowing convenient manipulations of the tree structure.
*   **Node Height:** The height `h` of a node is the sum of the edge lengths from the root to the node.
*   **Horizontal Spacing:** Nodes are equally spaced based on an in-order traversal of the tree.
*   **Node Order:** Left children will have smaller numbers, and the right larger numbers than their parents.
*   **Proposal:** The algorithm randomly shifts nodes up and down, changing the heights, and therefore changing the topology.
*   **Leaf Reordering:** Additional mechanisms are also included, such as leaf reordering.
