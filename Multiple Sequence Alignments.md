**1. Multiple Alignment Algorithms**

*   **What it is:** Imagine you have several sequences (like strings of DNA, RNA, or protein letters). Multiple alignment is like arranging these sequences in rows so that similar parts line up in columns. Think of it as finding the best way to compare many things at once.
*   **Why it's useful:** When comparing multiple sequences, you get more information than just comparing two sequences at a time. You can see what parts are similar across many sequences (maybe that's where the important functions are), what parts differ, and can even infer how these sequences might have changed over time (evolution).
*   **Gapped Sequences:** The aligned sequences will have "gaps" (represented as `-` or spaces) inserted to make the similar parts line up. This accounts for insertions or deletions that occurred during evolution.
*   **Formal Definition:** Given sequences  `X¹, X², ..., Xⁿ`, a multiple alignment creates new gapped sequences  `X̃¹, X̃², ..., X̃ⁿ`. These new sequences are created by adding gaps to the original sequences. They all have the same length, meaning `|X̃¹| = |X̃²| = ... = |X̃ⁿ|`.

**2. The Alignment Hyper-cube**

*   **What it is:** Imagine a cube (3D). Now think of a 4D hypercube, a 5D, or even higher dimensions. The alignment hypercube is similar. It is a multi-dimensional grid where each dimension represents one of the sequences that you are trying to align.
*   **Purpose:** Each corner or point (node) in this hypercube represents a specific state of aligning the sequences. A path through the hypercube is a specific alignment.
*   **How it works (Example):** Imagine you have 3 sequences, VSN, SNA, and -AS (using hyphens for alignment). In the hypercube you would have 3 dimensions. The axes would each represent the length of the sequences being compared. Each node is a possible position along this axes. A path from the beginning to the end, shows how the sequences are aligned to each other and what gaps have been added.
*   **Best Alignment:** The best multiple alignment corresponds to the "best" path through this hypercube. "Best" is usually defined by a score, we will talk about that later.

**3. Dynamic Programming Solution**

*   **What it is:** Dynamic programming is a way to solve a complex problem by breaking it down into simpler subproblems. Imagine you want to find the shortest path from A to C. Instead of trying every possible route, you first find the shortest paths to the intermediate points on the way to C.
*   **In the context of alignments:** In this method we store the scores in a table. The subproblems here are the subalignments, which are the alignments of the first *j1* characters of `X¹`, the first *j2* characters of `X²`, all the way up to the first *jr* characters of `Xr`. So, each step considers a different part of each of your input sequences.
*   **Function S:**  `S(j₁, j₂, ..., jᵣ)` represents the best score for aligning prefixes (the first *j* letters of each sequence). Each *jᵢ* indicates how far we have progressed through the *i*-th sequence.
*   **Finding S:** You calculate `S(j₁, j₂, ..., jᵣ)` by taking the best score from neighboring states in the hypercube (where neighbors means that you advanced by one character on at least one sequence). The neighboring states are found by considering all the possible values of `ε = (ε₁, ..., εᵣ)`, where each `εᵢ` can be either 0 or 1, but they can't be all zero.
*   **How to Find Neighbors:**  `ε` is a binary vector that tells you which sequences to advance in. If `εᵢ = 1`, advance in sequence *i*, meaning that we have aligned the *jᵢ*-th character of `Xⁱ`. If `εᵢ = 0` then a gap has been placed in sequence *i* meaning the last character in this sequence is used, but we don't advance in the sequence.
*   **Scoring Function (s):** The function `s(ε₁xⱼ₁, ..., εᵣxⱼᵣ)` is a scoring function, which tells us how good aligning the chosen characters is. For example, if the sequences have `A`, `T` and `-` in a certain position, then the scoring function will compute a score depending on the values of these characters. You add up the score for all aligned characters with the stored score `S(j₁ - ε₁, j₂ - ε₂, ..., jᵣ - εᵣ)`. The best alignment will be the one with the highest sum.
*   **Start and Calculation:** We start with the base case `S(0, 0, ..., 0) = 0`, meaning all sequences have no characters aligned, which has a score of 0. We build from here by applying the dynamic programming approach.

**4. Dynamic Programming Solution: Complexity**

*   **Complexity:** A fancy way to measure how hard the algorithm is to run, based on the input.
*   **Space Complexity:**
    *   The number of cells in the hypercube is `O(∏ᵢ₌₁ⁿ jᵢ)`, where *jᵢ* is the length of the *i*-th sequence. This means that the amount of memory needed depends on the lengths of the sequences being aligned.
    *   If all your sequences are the same length *n*, then the complexity becomes `O(nʳ)`, where *r* is the number of sequences being aligned. So the space the algorithm needs grows rapidly with the number of sequences.
*   **Time Complexity:**
    *   For every cell in this hypercube, we need to consider up to 2ʳ - 1 neighbor cells. The time it takes grows exponentially with the number of sequences. So, aligning more than a few sequences can be extremely slow.
    *   If the sequence lengths are *n*, then the time complexity becomes `O(2ʳnʳ)`.
*   **In Simple Terms:** Dynamic programming is an optimal solution, however, it becomes incredibly slow and memory-intensive as the number of sequences grows. This is why we need other approaches for aligning a lot of sequences.

**5. Scoring Metrics**

*   **What it is:** A way to measure the "goodness" of a multiple alignment. Different scoring schemes can change the alignment result.
*   **Important Factors:**
    *   **Position Specific:** Some positions within a sequence are more important or conserved than others. The scoring should take this into account, giving a high score for aligning important regions, and lower scores otherwise.
    *   **Phylogenetic Tree:** The sequences might not be completely independent. They likely have an evolutionary history, which could be represented by a tree. Ideally the scoring would know about these relationships.
*   **Ideal Scoring:**
    *   The perfect scoring method would be a complete probabilistic model that models all of the evolutionary events leading to your sequences. We don't have a model like this.
*   **Simplifying Assumptions:**
    *   **Position-Specific:** Treat each position independently and give scores for the characters that you have at this position, but ignore the underlying phylogenetic tree, so assume that all the sequences are independent.
    *   **Tree Model:** Account for the evolutionary tree and the relationships between the sequences, but treat each position independently.

**6. Multiple Alignments by Profile HMM Training**

*   **What is a profile HMM:** A profile HMM is a statistical model that learns from a set of aligned sequences and represents the different alignment positions with states. It is a way of representing a multiple alignment, and it captures which regions are conserved and which regions vary.
*   **How to use a profile HMM for MSA:**
    *   Imagine you've trained a profile HMM from a set of sequences. Now, you want to align a *new* set of sequences to that profile.
    *   Use the Viterbi algorithm to find the most probable path for *each* sequence through the HMM.
    *   Characters of the same match state will be aligned in columns.
*   **Key Points:**
    *   The alignment is position-specific due to the profile HMM, but the alignment assumes that the input sequences are independent.
    *   This is a way to get a quick multiple alignment if you already have a model.

**7. Computing the Multiple Alignment: Example**

*   **How does the HMM do it:** This example explains visually how sequences are aligned based on the match states of a Profile Hidden Markov Model. The main idea is that once you train your HMM, then when you see new sequences you find the best path using the Viterbi algorithm. If they are aligned in a column on the model (using a match state) then they are aligned to each other in the final multiple alignment.
*   **Example:** Imagine a HMM with the states `M1`, `M2`, `M3` and `M4`. Also include the insertion and deletion states `I`, and `D`. If sequence `x¹` is aligned to `M1`, and then passes through insertion, then alignes to `M2` and `M4`. Another sequence `x²` is aligned to `M1`, `M2`, `M3`, and `M4`. The final multiple alignment will show that the characters aligned to the same match states are also aligned to each other.

**8.  Multiple Alignments by Profile HMM Training (Revisited)**

*   **Parameter Estimation Problem:** The HMM needs to be trained first. For this, you need a set of aligned sequences, but this is difficult because usually, you just have unaligned sequences.
*   **Solution:** Use the EM (Expectation-Maximization) algorithm (Baum-Welch Algorithm).
*   **How EM works:** EM is an algorithm that is used when there is some missing information, but you still want to learn about your model's parameters. In this scenario, the information you are missing is the paths each sequence takes through the model. In the E-step (Expectation step) you estimate which paths are the best, and then in the M-step (Maximization step), you update the model based on your new path estimate. You do this iteratively and it usually converges to an optimal model.
*   **Key Points:**
    *   You start with some initial parameters of the HMM.
    *   EM relies on Forward and Backward probabilities to compute Expected Emission Counts (E_bl) and Expected Transition Counts (A_γt).
    *   Using these new expected counts you refine the HMM parameters until they converge.

**9. Simpler Multiple Alignment Algorithms**

*   **The Problem with HMMs and Dynamic programming:** HMMs are complex, and Dynamic Programming is slow. These methods can be computationally difficult to implement in practice.
*   **Sum of Pairs (SP) Score:** A simpler approach than HMMs, which consists of a sum of pairwise alignment scores. For every pair of sequences, you compute the score of the alignment, and then sum these scores up.
*   **SP Score (S(mⱼ)):** For every column mⱼ of the final alignment, the SP score will be the sum of the scores of the pairs of characters in this column. You sum all pairs of characters (mᵢ, mⱼ) for every column.
*   **Problem:** SP scores don't correspond to any kind of probabilistic model, it is just an heuristic. Log-odds scores would be better, however it is difficult to compute.

**10. Approximation Algorithms for MSA**

*   **Issue with SP:** Even with SP, MSA is still an NP-complete problem, which means that it takes a very long time to solve even with approximations.
*   **Costs instead of scores:** Instead of maximizing scores (s) we are now interested in minimizing costs (σ).
*   **Cost Function (σ):** σ(x, y) is the cost of aligning the character x with the character y (or a gap). It corresponds to the penalty for mismatching, or for adding a gap. A cost can be calculated from a score function by `σ(x,y) = exp(-λs(x,y))`.
*   **Assumptions:** It is important that these cost functions follow the rules. A gap with gap should have zero cost, also σ(x, y) should be equal to σ(y, x). Also the triangle inequality should hold:  `σ(x, y) ≤ σ(x, z) + σ(z, y)`.
*   **Problem Definition:** Given a set of sequences S, find an alignment that has the minimum total cost according to these rules.

**11. The Center Star Method for Alignment**

*   **What it is:**  A simple and fast approximation algorithm for MSA, instead of trying to align all the sequences simultaneously. The center star method picks one sequence and uses it as an anchor, and aligns all the sequences to that anchor.
*   **Center String:** It is the string Sc that minimizes the sum of all pairwise distances `D(Sc, Sj)`, so it minimizes the cost when compared with all other strings.
*   **Center Star:**  The alignment is constructed by choosing one string `Sc`, and aligning all the other strings against this single string. The center string is the central node of a tree. The other sequences are the leaf nodes of the tree.
*   **Approximation:** It is guaranteed that the alignment you get is no more than twice as worse than the best alignment possible (approximation ratio of two).
*   **Type-2 Approximation:** This method is type 2, meaning it is position independent (does not score based on positions) but considers an explicit evolutionary model (a tree).

**12. The Center Star Algorithm**

1.  **Find Center String:** Select `Sₜ` that minimizes `∑ᵢ₌₁ᵏ D(Sᵢ, Sₜ)`. Use this string as a first approximation to the final MSA `M`.
2.  **Add Sequences:** For all remaining strings `S` (that are not `Sₜ`), one by one you align them with the center string and merge the sequences to the alignment `M`. When aligning a new string `S`, you make sure that the alignment is optimal, you may need to add spaces in some positions, and these spaces need to be added to the other strings that are already in `M`.

**13. Progressive Alignment Heuristics**

*   **Problem with Center Star:** The center star method only uses one string as a guide. The progressive alignment algorithms use all the sequences, which is better.
*   **Idea:** Use a guide tree, instead of a center string, to decide how the multiple alignment should be built.
*   **Guide Tree:** The guide tree is a type of tree, where:
    *   **Leaves:** Each leaf is one of the sequences you want to align.
    *   **Inner Nodes:** These nodes represent a partial alignment between sequences.
*   **Building the alignment:** The progressive alignment algorithms works by first aligning the leaf nodes, and then combining the pairwise alignments until you have one global alignment on the root node.

**14. Progressive Alignment: ClustalW**

*   **What it is:** A very popular software program for multiple alignment that uses a progressive alignment algorithm.
*   **Algorithm:**
    1.  **Pairwise Alignments:** Compute all pairwise alignment scores for all pairs of sequences. Then, convert these scores into a distance matrix.
    2.  **Guide Tree:** Use Neighbor-Joining to build a tree based on the distances.
    3.  **Progressive Alignment:**  Starting from the leaves of the tree, combine the alignments following the tree. When aligning two nodes, either they can be sequences, a sequence and a profile, or two profiles.
*   **Features:**  Uses different scoring matrices, gap penalties and weighting schemes.
