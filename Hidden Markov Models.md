**1. Caveats of Optimal Alignment Algorithms**

*   **The Issue:** When comparing two sequences, algorithms always find *an* optimal alignment, but this alignment isn't always meaningful.
*   **Not Unique:** The "best" alignment isn't always the only one. There might be several alignments with similar scores, which are difficult to tell apart.
*   **Biologically Relevant Alignment:** The most important thing is finding the *biologically* relevant alignment, which shows how the sequences evolved from a common ancestor, instead of just a statistically optimal one.
*   **Probabilistic Treatment:** This highlights the need for a more probabilistic approach that can better account for these factors and find more meaningful alignments.

**2. Hidden Markov Models (HMMs)**

*   **What they are:** HMMs are powerful *probabilistic models* for analyzing sequences. They don't just give a score, they model how the sequences were generated.
*   **Typical Questions:**
    *   Does a given sequence (like a protein) belong to a specific family?
    *   If it belongs to a family, what can we say about its internal (secondary) structure?
*   **Power of HMMs:** HMMs can do *probabilistic sequence alignment*, which overcomes many limitations of traditional alignment methods.

**3. Markov Chains**

*   **What they are:** A Markov chain is a simple probabilistic model, which can be defined as a triplet, `(Q, P, A)`:
    *   `Q` is a finite *set of states*. Each state corresponds to a possible symbol in your sequences, from a alphabet `Σ`. For DNA, the alphabet is usually `Σ = {A, C, G, T}`.
    *   `P` (or `P(x₁)`), are *initial state probabilities*. It is the probabilities of starting your sequence in a specific state.
    *   `A` are the *state transition probabilities*. This tells you how probable it is to move between different states, denoted as `aₛₜ = P(xᵢ = t | xᵢ₋₁ = s)`. Here,  `aₛₜ` represents the probability of going to state *t* given that you are currently in state *s*.
*   **Key Property:**  Markov chains have a “memory of 1”. This means the probability of the next symbol `xᵢ` depends only on the previous symbol `xᵢ₋₁`, not on symbols further back in the sequence: `P(xᵢ|x₁, ..., xᵢ₋₁) = P(xᵢ|xᵢ₋₁)`.
*   **Memory Length 1:** It is said that these chains are random processes with memory length 1, which is a very useful simplification.

**4. Markov Chains (cont'd)**

*   **Joint Probability:** The joint probability of two random variables (RVs) *A* and *B* can be factorized as `P(A, B) = P(B|A)P(A)`.
*   **Total Probability of a Sequence:** The total probability of a sequence `X = (x₁, x₂, ..., xₗ)` is the product of the probabilities at each position given the previous position:
    `P(X) = P(xₗ|xₗ₋₁, ..., x₁) * P(xₗ₋₁|xₗ₋₂, ..., x₁) * ... * P(x₁)`
    ` = P(xₗ|xₗ₋₁) * P(xₗ₋₁|xₗ₋₂) * ... * P(x₁)`
    ` = P(x₁) ∏ᵢ₌₂L aₓᵢ₋₁ₓᵢ`.

    Here, `aₓᵢ₋₁ₓᵢ` represents the probability of transitioning from symbol `xᵢ₋₁` to `xᵢ`
*   **Silent States:** In order to represent the beginning and end of the sequence we add "silent states" which do not appear in the sequence:
    *   **Begin State:** `x₀ := B`, meaning `P(x₁ = s) = a_Bs` where `a_Bs` is the transition from the Begin state to a new state *s*.
    *   **End State:** `xₗ₊₁ = E`, meaning `P(E|xₗ = t) = a_tE` where `a_tE` is the transition from the last state *t* to the End state.
    *    **Total Probability with silent states** The probability of the complete sequence with the silent states is given by `P(X) = ∏ᵢ₌₁L₊¹ aₓᵢ₋₁ₓᵢ`.

**5. A Markov Chain for DNA Sequences**

*   **Visual Representation:** Here we have a graphical representation of a Markov chain, for the nucleotides in DNA (`A`, `C`, `G`, and `T`).
*   **Nodes and Arrows:**
    *   Each node (`A`, `C`, `G`, `T`) is a state. The starting point is `B` (Begin) and the end is `E` (End)
    *   The arrows between states are transitions with transition probabilities that are not shown here for clarity, but are part of `A`.

**6. Example: CpG Islands**

*   **What they are:** CpG refers to a *dinucleotide* where the letter `C` is followed by a `G`. They are short stretches of DNA with a higher frequency of CG.
*   **Rare in DNA:** Typically, `CpG` dinucleotides are rare compared to the probabilities of `C` and `G` occurring independently (`P(C)P(G)`).
*   **Frequent in Islands:** There are some regions where `CpG` dinucleotides are more frequent, which are called *CpG islands*.
*   **Significance:** These CpG islands are often located in important areas of the genome, near the starting points of genes.

**7. Example: CpG Islands (cont'd - Visual)**

*   **Visual Representation:** This slide shows example DNA sequences where the different CpG islands can be seen in the text (highlighted in red).
*   **Left:** The left shows a sequence with a high density of CpG sites. This a promoter (start) region, which includes a start codon `ATG`.
*   **Right:** The right shows a sequence with a lower density of CpG sites, which is more representative of normal genomic regions where the DNA is usually methylated.

**8. A Markov Model for CpG Islands**

*   **Problem:** To identify CpG islands within a DNA sequence.
*   **Input:** A short DNA sequence, `X = (x₁, ..., xₗ)` where `xᵢ` can be `A`, `C`, `G`, or `T`.
*   **Question:**  Decide whether the sequence X is part of a CpG island.
*   **Two Markov Chains:** Use two different Markov chains: one for sequences *inside* CpG islands ("+" model) and another for sequences *outside* CpG islands ("-" model).
*   **Transition Probabilities:**  `a⁺/₋ₛₜ` represents the transition probabilities *inside* ("+") or *outside* ("-") CpG islands.
*  **Score function:** The scoring function is defined as the log-odds ratio of the probability that a sequence came from inside a CpG islands (using `a⁺`) or from outside a CpG island (`a⁻`). `score(X) = log P(X|CpG island)/P(X|non CpG island) = ∑ᵢ₌₁L log(a⁺ₓᵢ₋₁ₓᵢ/a⁻ₓᵢ₋₁ₓᵢ)`.
* **High Score:** A high score suggests that it is more likely that the sequence *X* is part of a CpG island.

**9. Estimating the Probabilities**

*  **Training Sample:** To estimate the transition probabilities of the Markov chains we use a training sample that has the label of whether the sequences are part of a CpG island or not.
*   **Counting:** The transition probabilities are estimated by counting the number of times that a letter *t* follows a letter *s*, `c⁺/₋ₛₜ`, inside ("+") or outside ("-") of CpG islands.
*   **Maximum Likelihood Estimation:** We calculate the frequencies (as fractions of the totals) `a⁺/₋ₛₜ = c⁺/₋ₛₜ / (∑ₜ c⁺/₋ₛₜ′)`.
*   **Two Tables:** The result are two transition probability tables: One for CpG regions and the other for non-CpG regions.

**10. Identifying CpG Islands: Experimental Results**

*   **Results of Training:** A real world example of the transition probabilities (for `A`, `C`, `G` and `T`) after training the model with a human DNA dataset.
*   **Two Markov Chains:** Two tables show the estimated transition probabilities for the "inside CpG island" model (`+`) and the "outside CpG island" model (`-`).
*  **Length Normalized Scores:** We present the length normalized score of the sequences (in bits). The score is normalized by dividing by the length of the sequence.
    `score(X) = 1/L log(P(X|CpG island)/P(X|non CpG island)) = 1/L ∑ᵢ₌₁L log(a⁺ₓᵢ₋₁ₓᵢ/a⁻ₓᵢ₋₁ₓᵢ)`.

**11. Identifying CpG Islands: Experimental Results (Visual)**

*   **Histograms:** This is a plot showing two histograms based on experimental results:
    *   The dark grey histogram represents sequences from actual CpG islands, using their length-normalized score.
    *   The light grey histogram shows sequences that are *not* in CpG islands, using their length-normalized scores.
*   **Separation:** There is a clear separation, meaning that it's easy to tell apart sequences inside CpG islands from sequences outside. Sequences in CpG islands are usually associated with scores greater than 0.

**12. Locating CpG Islands**

*   **Problem:** Find *where* the CpG islands are in a *long* DNA sequence.
*   **Input:** A long DNA sequence, `X = (x₁, ..., xₗ)`, where `xᵢ` can be `A`, `C`, `G`, or `T`.
*   **Question:** Find the locations of all CpG islands along *X*.
*   **Naive Solution:**  A simple strategy is to extract all possible subsequences of a fixed size *l* from *X*. We call these subsequences `Xᵏ`.
    *   Then, we calculate a score for each `Xᵏ`, using `score(Xᵏ)`.
    *   Subsequences with positive scores are considered as potential CpG islands.
*   **Problem:** There is not any information about the length of the CpG islands. Also we need to specify the parameter *l* of the size of the substring, which is also unknown.
    *   **l too large:** Islands can be smaller that *l*. Also, scores may not be large enough to discriminate.
    *   **l too small:** Small windows don't provide enough information to discriminate.

**13. A Unified Model for Locating CpG Islands**

*   **The Idea:** Instead of using two separate chains, use a *single unified model*. This model is composed of the *two chains* for inside and outside islands, as well as transitions between them with small probabilities.
*   **Visual Representation:** You can see the two sets of states (`A⁺`, `C⁺`, `G⁺`, `T⁺` and `A⁻`, `C⁻`, `G⁻`, `T⁻`), and all transitions between them. The probabilities of transitioning between states are not shown.

**14. Hidden Markov Models (HMMs - Revisited)**

*   **Formal Definition:** An HMM is a triplet M = (Σ, Q, Θ):
    *   `Σ` is the alphabet (e.g., nucleotides or amino acids).
    *   `Q` is the set of *hidden states*. They are hidden because we don't observe them, only the symbols that they emit.
    *   `Θ` is a set of probabilities that define the HMM:
        *   *State transition probabilities*, `aₖₗ` are the probabilities of transitioning from state `k` to state `l`. `aₖₗ = P(πᵢ = l | πᵢ₋₁ = k)`.
        *   *Emission probabilities*, `eₖ(b)` are the probabilities that state `k` emits the symbol `b`. `eₖ(b) = P(xᵢ = b | πᵢ = k)`.
*  **Path:** A path `Π = (π₁, ..., πₙ)` is a sequence of hidden states.
*   **Markov Property:** The probability of moving to a given state depends only on the previous state.

**15. Hidden Markov Models (cont'd)**

*   **Probability of a Sequence:**  The probability of a sequence `X` was generated by model M using the path `Π`: `P(X, Π) = aπ₀π₁ ∏ᵢ₌₁L eπᵢ(xᵢ) aπᵢπᵢ₊₁`
    *  `π₀` represents the Begin state, and `πₙ₊₁` represents the End state.
    * The formula calculates the probability of a sequence, while following a particular path.
*   **Visual Example:** A graphical representation showing the states Begin and End, and a sequence of states (colored in magenta and light brown), and how the sequence `AACTGGCGAATTCAGTCGAGGGTTAC` is emitted.

**16. Example 1: An HMM for Detecting CpG Islands**

*   **Goal:** Here we re-visit the CpG island example, but using an HMM, instead of Markov chains.
*   **State Space:** In this HMM there are 8 states. Four states that emit DNA sequences in CpG regions `A⁺`, `C⁺`, `G⁺`, `T⁺`, and four states that emit DNA sequences outside CpG islands `A⁻`, `C⁻`, `G⁻`, `T⁻`.
*   **Emission Table:** Each state is associated to an emitted symbol. States `A⁺` and `A⁻` emit `A` with probability 1. States `C⁺` and `C⁻` emit `C` with probability 1 and so on. All states have zero probability of emitting any of the other nucleotides, so `P(xᵢ = A | πᵢ = A⁺/⁻) = 1`, and `P(xᵢ = A | πᵢ = {C,G,T}⁺/⁻) = 0`.
*  **Key difference between HMM and Markov chains** In this HMM there is no longer a one-to-one correspondence between states and the symbols. Each state can only emit a specific symbol, but the path (the state sequence) that generated the symbols is *hidden*.

**17. Example 2: Modeling a Dishonest Casino**

*   **Scenario:** This example represents a simple, but instructive problem: modeling a casino with a dishonest dealer, who sometimes switches to a loaded die.
*   **Fair Die:** Most of the time, the dealer uses a fair die, so all the numbers are equally probable (1/6).
*  **Loaded Die:** Sometimes he uses a loaded die where the 6 is more probable (`P(6) = 0.5`) and all other numbers have the same small probability (`P(1 to 5) = 0.1`).
*  **Transition Probabilities:** The dealer switches from the "fair" state (F) to the "loaded" state (L) with a probability of `a_FL = 0.05`, and he switches back to the "fair" state with probability `a_LF = 0.1`
* **Visual Representation:** A graphical representation of the model. Here there is a transition diagram between the Fair state and the Loaded State. Also, there is an emission probability for each of the 6 sides of the die within the Fair state, and within the Loaded state.

**18. The Decoding Problem**

*   **Observed Sequence:** You have a sequence of emitted symbols `X = (x₁, ..., xₗ)`.
*   **Hidden Path:** You don't know which sequence of states, `Π = (π₁, ..., πₗ)`, generated that sequence. The path is hidden.
*   **CpG Islands:** if we knew the path we could see where the "+" states were, and therefore which parts of the sequence are part of CpG islands.
*   **The Decoding Problem:** Find the *most probable* state sequence (`II*`), given the observed sequence `X`.
     `II* = argmax_Π P(X, II)`.

**19. The Viterbi Algorithm**

*  **Goal:** The goal is to find the most probable path that an HMM took to generate a sequence of symbols.
*   **Notation:**  `vₖ(i)` is the probability of the *most probable* path for the *prefix* `(x₁, ..., xᵢ)` that *ends* in state `k`.
*  **Matrix:** We store all the `vₖ(i)` values in a matrix *V*. Each entry in the matrix `Vₖᵢ = vₖ(i)`.

**20. The Viterbi Algorithm: Main Idea**

*  **Recursive Approach:** The viterbi algorithm follows a recursive approach.
* **Assumption:** You assume that you already know `vₖ(i-1)`, for all states *k*. These values are the probabilities of the best path for the prefix `(x₁,...,xᵢ₋₁)` ending in the state `k`.
* **Key Idea:** You can compute `vₗ(i)` as the product of the emission probability of state `l` given `xᵢ` (`eₗ(xᵢ)`) times the best score ending on any of the previous states (`maxₖ(vₖ(i-1)aₖₗ)`).
* **Diagram:** You can see a visual representation where there are three different states on the previous step `v₁(i-1)`, `v₂(i-1)`, `v₃(i-1)`, and they transition to the next state *l* at position *i*.

**21. The Viterbi Algorithm: Recursion**

*   **Formalization** This slide presents the mathematical steps of the Viterbi algorithm.
*   **Initialization:**
    *   `v_B(0) = 1`: Start from the Begin state `B` with probability 1.
    *   `v_k(0) = 0` for all other states `k`, because it is not possible to start from those states.
*   **Recursion:** Calculate `vₗ(i)` for all the states *l*:
    *   `vₗ(i) = eₗ(xᵢ) * maxₖ(vₖ(i-1)aₖₗ)`.  This is the multiplication of the emission probabilities with the probability of the best previous state *k*.
     *  `ptrₗ(i) = arg maxₖ(vₖ(i-1)aₖₗ)`. Store the previous state *k* which resulted in the max value. This value will be used for traceback.
*   **Termination:**
    *   `P(X, II*) = maxₖ(vₖ(L) aₖₑ)`. The probability of the best path is obtained by calculating the best path that goes to the end state `E`.
    *    `πₗ* = arg maxₖ(vₖ(L)aₖₑ)`. Here we compute what is the best state (and store it in `πₗ*`).
*   **Traceback:**
    *  The traceback step traces the pointers backwards to compute the best state sequence from the termination step `πᵢ₋₁* = ptrₗ(πᵢ*)`.

**22. The Viterbi Algorithm (cont'd)**

*   **Problem: Underflow** The Viterbi algorithm involves multiplying a lot of small numbers, which can lead to computer underflows.
*   **Solution: Log Scores**  Replace probabilities with their logarithms to transform products into sums:
     `vₗ(i+1) = log(eₗ(xᵢ₊₁)) + maxₖ{vₖ(i) + log(aₖₗ)}`. This is equivalent, but it avoids underflow.
*   **Time Complexity**
    *   Calculate `O(|Q|L)` values (cells of the matrix *V*), where Q is the set of states.
    *   For every cell, we must perform O(|Q|) operations. So the overall time is O(|Q|² L).
*   **Space Complexity**
    *   The memory requirement to store all values is O(|Q|L).

**23. A Dishonest Casino (cont'd - Results)**

*   **Experiment:** The dishonest casino HMM model was used to generate 300 random die rolls.
*   **Viterbi Output:** The Viterbi algorithm was applied to this sequence to compute the most likely path. The most probable path was extracted by Viterbi, and they are presented here.
*   **Representation:** You see four experiments. For each experiment you see the "Rolls", the "Die", which represents the true state when generating the sequence (whether the "loaded" or "fair" die was used), and the "Viterbi" output which represents what the model predicted was the most probable path. The model does pretty well at identifying which parts were created by the loaded dice.

**24. The Total Probability of a Sequence**

*   **Markov chains vs HMM** The markov chain approach gives you the probability of a sequence, but it does not work well if there is more than one possibility, and it assumes that you know which path generated the sequence. HMMs, on the other hand, allow a single state to generate multiple outputs.
*   **HMM problem** The problem with HMMs is that the probability of a sequence given a path `P(X, II)` is easy to compute, but the total probability of a sequence `P(X)` is more difficult.
*   **Adding all paths:** To compute `P(X)` we need to add the probability of *all* possible paths that might have produced `X`. `P(X) = ∑Π P(X, II)`.
*  **Input and Question:**  Here we define what is the goal when you are calculating the total probability.
    * Input :  A hidden Markov model and a sequence X over some alphabet, and the goal is to compute `P(X)`.

**25. The Total Probability of a Sequence (cont'd)**

*   **Problem: Exponential Paths**  The number of all possible paths grows *exponentially* with the length of the sequence `X`, which means that adding all paths is not feasible for long sequences.
*   **Approximation using Viterbi:** An approximation would be to compute the *most probable* path (`II*`), using the Viterbi algorithm, and to consider that the probability of this path is a good approximation for the overall probability:  `P(X, II*) = aπ₀π₁*  ∏ᵢ₌₁L eπᵢ*(xᵢ) aπᵢ*πᵢ₊₁*`.
*   **Assumption:** The method assumes that this most probable path is the only significant path, which is not always correct.
*   **Better approach:** A better way to calculate this probability, is by replacing *maximization* with *summation* in a dynamic programming approach, which is called the *forward algorithm*. The forward algorithm adds the probabilities of all possible paths.
*   **Correctness of Viterbi** To assess how correct the Viterbi path is, we can evaluate the ratio `P(II*|X) = P(X,II*) / P(X)`. This tells us if the Viterbi output is a good representation of the best path, given the data.

**26. The Forward Algorithm**

*  **Goal:** To efficiently compute the sum of probabilities of all the paths through the model.
*  **Forward variable:** `fₗ(i)` is defined as the probability of emitting a prefix `(x₁,...,xᵢ)` and ending up in state `πᵢ = l`.
*   **Recursive computation:** The values `fₗ(i)` are computed recursively starting from the beginning. The value of the last state in `xᵢ` is computed using all the previous values using `fₗ(i) = eₗ(xᵢ) ∑ₖ fₖ(i-1)aₖₗ`.  We are summing the probabilities of paths going to different states *k*, but multiplying by the transition probability from *k* to *l* and by the probability of emitting `xᵢ` on state `l`.
*   **Diagram:**  You can see a visual representation of the forward algorithm. You can see how the probability of each of the previous states `f₁(i-1)`, `f₂(i-1)`, `f₃(i-1)`, is used to compute the value at the new state *l*.

**27. The Forward Algorithm (cont'd)**

*   **Mathematical Formulation:** This slide presents the formal steps for the forward algorithm.
*   **Initialization:**
    *   `f_B(0) = 1`: Initialize the probability to 1 for being at the start (begin state `B`).
    *   `f_k(0) = 0`:  All other states are initialized to 0.
*   **Recursion:** Compute `fₗ(i)` for all states, by doing the following for every position *i* and all the states `l`:
    * `fₗ(i) = eₗ(xᵢ) ∑ₖ fₖ(i-1)aₖₗ`
*  **Termination:** Calculate the final probability as a sum over all states: `P(X) = ∑ₖ fₖ(L) aₖₑ`.

**28. Posterior Decoding**

*   **Viterbi:**  Finds the most probable *path* through the model, which is optimal for alignment.
*   **Forward:** Computes the *total probability* of a sequence, by adding the probabilities of all possible paths.
*  **Posterior probabilities:** The next step is to compute the probability of a state at a particular point, `i`, of a sequence.
*   **Posterior Probability** `P(πᵢ = k | X)`: the probability that the observation *xᵢ* came from state *k*, *given* the entire observed sequence *X*.

**29. Posterior Decoding (cont'd)**

*   **Posterior Decoding Problem:**
    *   **Input:** An HMM `M` and a sequence `X`. The generating path `II` is unknown.
    *   **Question:** For each position `1 <= i <= L`, and for each state `k`, compute `P(πᵢ = k | X)`.
*   **Formula:**
    *   `P(πᵢ = k | X) = P(X, πᵢ = k) / P(X)`.
*   **Prefix and Suffix:** We separate the sequence in two parts.  The part that occurs before `xᵢ`, `prefix = (x₁, ..., xᵢ)`, and the part that occurs after `xᵢ`, `suffix = (xᵢ₊₁, ..., xₗ)`.
*   **Memory Length 1:** We can simplify `P(X, πᵢ = k)` by using the markov property, as follows.
   * First, you have that `P(A, B) = P(A) P(B|A)`.
   * Then, `P(X, πᵢ = k) = P(x₁,...,xᵢ, πᵢ = k) P(xᵢ₊₁,...,xₗ | x₁,...,xᵢ, πᵢ=k)`
  * By using the Markov property, you have that   `P(X, πᵢ = k) = P(x₁,...,xᵢ, πᵢ = k) P(xᵢ₊₁,...,xₗ | πᵢ = k)`.
  * This can be represented using the notation from the forward and backward algorithms: `P(X, πᵢ = k) = fₖ(i) * bₖ(i)`.

**30. The Backward Algorithm**

*   **Suffix probability:** The backward variable `bₖ(i)` represents the probability of emitting the *suffix* `(xᵢ₊₁, ..., xₗ)` given that you are at state `πᵢ = k`.
* **Diagram:** You can see a visual representation where state *k* at position *i* can emit the suffix with the states `b₁(i+1)`, `b₂(i+1)`, and `b₃(i+1)`. Each of these states is connected to the previous state *k*.
*   **Recursion:**  The values for `bₖ(i)` are calculated recursively, by summing over the transitions and emissions to the next states. The new value is computed as `bₖ(i) = ∑ₗₑQ aₖₗ eₗ(xᵢ₊₁) bₗ(i+1)`.

**31. The Backward Algorithm (cont'd)**

*  **Formal Definition**
*  **Initialization:**  Start from the end of the sequence. In the end we are interested in a transition to the End state `E`, so the initialization is `bₖ(L) = aₖₑ`, for all states *k*.
*  **Recursion:** The recursion to compute the backward probability is `bₖ(i) = ∑ₗₑQ aₖₗ eₗ(xᵢ₊₁) bₗ(i+1)`.
*  **Termination:** The total probability of the sequence `P(X)` can also be computed using the backward variables as `P(X) = ∑ₗₑQ a_Bₗ eₗ(x₁) bₗ(1)`.

**32. A Dishonest Casino (cont'd - Visual)**

*   **Posterior Probabilities:** This slide shows the posterior probabilities of the state at each position of the casino model.
*  **Interpretation:** The probability of being at the "fair" state, `P(fair)`, is calculated at every position and is plotted on the graph. The colored regions are showing the times when the loaded die was used.
*   **Usefulness:** Here we can see that there are some locations where the model is more certain than others.

**33. Uses for Posterior Decoding**

*   **Many Similar Paths:** If multiple paths have very close probabilities, we can't rely solely on the Viterbi path, and it becomes useful to look at the posterior probabilities instead.
*   **Alternative Path (II\*\*):** An alternative path (II\*\*) that you can calculate by selecting each state kᵢ at position *i* that has the highest posterior probability  `II\*\* = argmaxₖ{P(πᵢ = k | X)}`.
*  **Caveats:**
   * The path II\*\* is not a particularly likely path as a whole (it is locally optimal), so the sequence might not be a likely path in the HMM.
    * II\*\* might not even be a valid path because transitions between states might not be possible.

**34. Uses for Posterior Decoding (cont'd)**

*   **Derived Property:**  Sometimes you are not interested in the path itself but in a particular property of it, defined as `g(k)`. The expectation of `g(k)` is  `G(i | X) = Ep(πᵢ = k | X) [g] = ∑ₖ P(πᵢ = k | X) g(k)`.
*   **Example:**
    *   If `g(k)` is 1 if the state `k` belongs to a subset of states `S`, and 0 otherwise. Then `G(i|X)` is the probability that a state in S emitted `xᵢ`.
*   **CpG Example:** In the CpG model, `g(k)` could be 1 if the state is an `A⁺`, `C⁺`, `G⁺` or `T⁺` and 0 if it's `A⁻`, `C⁻`, `G⁻` or `T⁻`. Then `G(i | X)` becomes the posterior probability that `xᵢ` comes from a CpG island.

**35. Parameter Estimation**

*   **Recall:** HMM is a triplet `M = (Σ, Q, Θ)`. `Θ` are the probabilities: the transition probabilities `aₗₘ` and the emission probabilities `eₗ(b)`.
*  **Problem:** You want to estimate the model parameters `Θ` when they are unknown.
* **Training Sequences:** You use `n` independent training sequences `X = {X₁, ..., Xₙ}` and the corresponding generating paths `Ψ = {Π₁, ..., Πₙ}`. These sequences are often labelled (for example: the paths might be known for the data you are using to train the model).
*   **Total Likelihood:** For all the `n` training sequences, the likelihood is `P(X, Ψ | Θ) = ∏ᵢ₌₁ⁿ P(Xᵢ, Πᵢ | Θ)`. And the probability `P(Xᵢ, Πᵢ | Θ)` can be expressed as `∏ₖ₌₀ⁿ  ∏ᵢ₌₀L eπᵢ(xᵢᵏ) aπᵢᵏπᵢ₊₁ᵏ`, where `k` represents which training sequence we are looking at.

**36. Parameter Estimation: Known State Paths**

*   **Maximum Likelihood:** The goal is to estimate `Θ` by maximizing the total likelihood. You want the model that best explains the observed data.
     `Θ̂ = argmax_Θ P(X, Ψ | Θ)`.
*  **Assumption:** Assume that the paths of the training sequences are already known (labelled data).
*  **Counts:** Count the number of times that the *b*-th observation symbol is emitted from the *l*-th state, `Eₗ(b)`. And count the number of transitions from state *m* to state *l*, `Aₗₘ`.
*   **ML Estimates:** The maximum likelihood (ML) estimates for the parameters `Θ` are simply the observed frequencies: `êₗ(b) = Eₗ(b) / (∑_b' Eₗ(b'))` and `âₗₘ = Aₗₘ / (∑_l' A_l'm)`.

**37. Parameter Estimation: General Case**

*   **Unlabelled Data:** In real-world data, the paths are usually unknown.
*   **EM Algorithm:** You can use an Expectation-Maximization (EM) Algorithm. This algorithm works as follows:
    *   **E-step (Expectation):** Compute the *expected* counts of emissions `Eₗ(b)` and transitions `Aₗₘ`, considering *all* possible paths `Ψ`:
        * `Eₗ(b) = ∑Ψ P(Ψ | X, Θ) E_Ψ(b)`
        *  `Aₗₘ = ∑Ψ P(Ψ | X, Θ) A_Ψlm`
    *  **M-step (Maximization):** Use the expected counts to get new estimates for `Θ`, by maximizing the likelihood:
        *   `êₗ(b) = Eₗ(b) / (∑_b' Eₗ(b'))`
        *   `âₗₘ = Aₗₘ / (∑_l' A_l'm)`
*   **Iterative Nature:** Notice that `P(Ψ|X, Θ)` (the probabilities that we are calculating) depends on the model parameters, `êₗ(b)` and `âₗₘ`, so this process has to be done iteratively until convergence.

**38. Parameter Estimation: General Case (cont'd)**

*   **E-step in detail:**  The forward and backward algorithm are used to compute the expected counts, by using the probability `P(Ψ|X,Θ)`. The value of `Eₗ(b)` is calculated as a sum of all training sequences, and it is the sum of probabilities over all positions *i* that emit the symbol *b*, for state *l*. The probability at the given position *i* is given by `fₗ(i)bₗ(i)/P(Xʲ)`. So, the formula is:
    `Eₗ(b) = ∑ⱼ  1/P(Xʲ)   ∑{i|xᵢ=b} fₗʲ(i)bₗʲ(i)`. Here, `j` is an index over the training sequences.
*   **Transition Counts** The transition counts are calculated in a similar way:  `Aₗₘ = ∑ⱼ 1/P(Xʲ) ∑ᵢ fₗʲ(i) aₗₘ eₘ(xᵢ₊₁)bₘʲ(i+1)`. Here, `j` represents the training sequences.

**39. Identifying Prokaryotic Genes**

*   **What are Prokaryotes:** These are simple cellular organisms (like bacteria or archaea) that *do not* have a nucleus. In contrast, Eukaryotes do have a nucleus.
*   **Visual Representation:** This slide provides a visual contrast between prokaryotic and eukaryotic cells:
    *   **Eukaryote:** The eukaryotic cell has a nucleus, a mitochondria, ribosomes, nucleolus, and a membrane-enclosed nucleus.
    *   **Prokaryote:** The prokaryotic cell is less complex, it has a nucleoid, ribosomes, a capsule (in some prokaryotes), a flagellum, and a cell membrane and cell wall.

**40. Identifying Prokaryotic Genes (cont'd)**

*  **Domains of Life:** All life is divided into three domains: Bacteria, Archaea, and Eukaryota. Prokaryotes consist of Bacteria and Archaea.
*   **Visual Representation:** A diagram showing how the domains of life relate to each other.

**41. Identifying Prokaryotic Genes (cont'd)**

*   **Prokaryotic Genes:** Genes in prokaryotes have a simple structure: a start codon followed by codons that encode amino acids followed by a stop codon.
*   **Codons:** A codon is a sequence of 3 nucleotides (triplets), which corresponds to an amino acid. There are 61 codons that encode for amino acids, and 3 stop codons (that terminate the protein).
*   **Visual Representation:**  A visual representation shows how the genes are arranged on the DNA and how they are read as codons. The structure is such that a start codon initiates translation to an amino acid, which occurs by reading the following codons, until a stop codon.

**42. Identifying Prokaryotic Genes (cont'd)**

*  **Gene Candidates:** The simplest way to look for genes is to look for stretches of DNA with the right structure. Such candidates are called *open reading frames* (ORFs).
*   **Problem:** There are many more ORFs than real genes, because they can be generated by chance.
*   **Idea:** Build an HMM to distinguish real genes from spurious ORFs. The idea is to encode all the possible codons (64) as different characters, and train an HMM. This HMM has a similar architecture to the CpG islands HMM.
    *  `+` states represent genes.
    *  `-` states represent non coding ORFs (NORFs).
* **Experimental Data:** Data from the E. coli genome, including 1100 genes and around 30000 NORFs, randomly split for training and test purposes.

**43. Identifying Prokaryotic Genes (cont'd - Visual)**

*   **Histograms:** A visual representation of two histograms:
    *   The grey histogram is the log-odd score of all non-coding ORFs (NORFs) (negative values).
    *   The black line shows the log-odd scores for all the real genes (positive values).
*   **Interpretation:** There is a clear separation between the two histograms, so the HMM works to separate real genes from spurious ORFs.

**44. Hidden Markov Models: Pair HMMs for Sequence Alignment**

*   **New Section:**  This slide introduces a new concept: Pair HMMs for sequence alignment.
*   **Visual Representation:** You can see a visual representation of the state space in two different pair HMMs:
    *  **Left**  is the representation of a standard alignment algorithm that has a Match state `M`, Insertions in x `Ix`, and insertions in y `Iy`.
   *   **Right** shows a corresponding representation using a probabilistic framework: a Match state `M`, insertions in `x` (X), and insertions in `y` (Y).
 *   **Emissions:** For example in `M`, you see the probability that the state emits characters x and y (`p_xiyi`), which corresponds to the match states of traditional alignment.
* **Key difference** In a pair HMM the goal is to generate pairs of aligned sequences, where the emission is not a single character, but a pair of characters or a character and a gap.

**45. Pairwise Alignment using HMMs**

*   **Markov Property:** The states of the HMM satisfy the Markov property, where transitions only depend on the last state.
*   **Emitted Symbols:** The HMM emits *sequences of symbols*. However, unlike previous HMMs, here the emitted sequence is a *pair* of symbols, which correspond to aligned characters or to characters and gaps.
*  **Inference:** Because we only see the emitted sequences (the aligned sequences), the path through the HMM is unknown and we have to use inference algorithms (such as Viterbi) to get an estimate of the most likely path.
*   **Key Point:** Knowing the most probable path, allows you to analyze the *internal structure* of the string (which regions are matches, which ones are gaps, etc).

**46. Pair HMMs for String Alignment**

*   **Sequence Alignment:** HMMs can be used for alignment. A pair HMM emits pairs of aligned characters, instead of a single character.
*   **Pair HMMs:**  A special type of HMM where the states emit a *pair* of symbols.
*   **FSA to Pair HMM:** Pair HMMs are derived from Finite State Automata (FSA).
*   **Emission Probabilities:** In a Pair HMM you have emission probabilities for all the states: `M` emits pairs of characters (`xᵢ <> yⱼ`) with probability `p_xᵢyⱼ`, the insertion state `X` emits the character `xᵢ` followed by a gap, with probability `q_xᵢ`, the insertion state `Y` emits a gap followed by the character `yⱼ` with probability `q_yⱼ`.
*   **Transition Probabilities:** Also, it is important to define all the transition probabilities between the states, so all the probabilities that go out of a given state have to sum to 1.
*   **Begin and End States:** The model needs to include begin and end states to define the alignment boundaries.

**47. FSAs and Pair HMMs (Visual)**

*   **Visual Representation:** Three different visualisations of Pair HMMs. These visualisations represent a Finite State Automata (FSA) and Pair HMMs.
    *   **Top left**: A classic representation of a pairwise alignment using states: `M`, `Ix` and `Iy`. Here transitions go from M to Ix or Iy or back to M. This would be equivalent to a gap-affine algorithm (such as Smith-Waterman).
    *   **Top Right**: A visualization of a Pair HMM with a match state `M`, with an insertion to x state (`X`) or an insertion to y state (`Y`). Each of these states has self-transitions as well.
    *   **Bottom:** A visualization of a full model, including beginning and end states, with a match state (`M`) and insertion to x (`X`) or insertions to y (`Y`).
*  **Transition and Emissions:** The emissions and transitions are explicitly shown in the models.

**48. Recall: General Structure of Viterbi Algorithm**

*   **Revisiting Viterbi:** This slide is a review of the Viterbi algorithm to find the most likely path.
*   **Viterbi Variable:**  `vᵢ(i)` is the probability of the most probable path ending in state *l* after reading the first *i* characters in the sequence.
*   **Equation**  The value for the Viterbi variable is computed using the equation: `vₗ(i) = eₗ(xᵢ) * maxₖ(vₖ(i-1)aₖₗ)`.

**49. Pair HMMs: Generative Models for Alignments**

*   **Pair HMMs for Alignment:** This slide visualizes a pair HMM, where the goal is to generate pairs of sequences that are aligned.
*   **Visual Representation:** You see the usual match state `M`, which is connected to states that insert a character on `x` (state X) and inserts a character in `y` (state `Y`). Also there are self-transitions for each of these states.
*   **Viterbi variable:**  `v^(M/X/Y)(i,j)` is the probability of the most probable path for a *prefix* alignment ending in a specific state (`M`, `X`, or `Y`) at position *i* of sequence *x* and *j* of sequence *y*.

**50. Pair HMMs: Viterbi Algorithm (cont'd)**

*   **Initialization:**
    *   `vᴹ(0,0) = 1`: Start from the match state with probability 1.
    *    `vᴹ(0,j) = vᴹ(i,0) = 0` The other cells in M are initialized to 0.
    * Initialize `vˣ(0, j)` and `vʸ(i,0)` using random model.
*   **Recurrence:** You use dynamic programming to iteratively compute the variables for each position in x (*i*) and each position in y (*j*).
    *   `vᴹ(i, j) = pₓᵢyⱼ * max{(1 – 2δ – τ)vᴹ(i-1, j-1), (1 - ε - τ)vˣ(i-1, j-1), (1 - ε - τ)vʸ(i-1, j-1) }`
        The value for the match state (M), depends on the emission probabilities `p_xiyi` at that position, and the maximum score between all previous states that could transition to M.
*  **Graphical Representation** The slide also shows a graphical representation of the HMM used in this formula.

**51. Pair HMMs: Viterbi Algorithm (cont'd)**

*   **Viterbi equations:** (Continued from the previous slide) This slide shows the equations of how the Viterbi algorithm works when applied to pair HMMs. The values in the previous slide for the match state were `vᴹ(i, j)`. In this slide, the corresponding equations are presented for `vˣ(i,j)` and for `vʸ(i,j)`.
  *   `vˣ(i, j) = qₓᵢ * max{δvᴹ(i-1, j), εvˣ(i-1, j)}`. The values for the state `X` which corresponds to insertions in x, depend on how a path arrived to this state either from a match state M or from an insertion x state X, with probabilities `δ` and `ε`, respectively.
    *    `vʸ(i, j) = q_yⱼ * max{δvᴹ(i, j-1), εvʸ(i, j-1)}`.  The values for the state `Y` which corresponds to insertions in y, depend on how a path arrived to this state either from a match state M or from an insertion y state Y, with probabilities `δ` and `ε`, respectively.
*   **Visualizations:** In this slide you see visualizations of how the different states relate to each other, and the probabilities that control the paths within each region.

**52. Pair HMMs: Viterbi Algorithm (cont'd)**

*   **Termination** We can get the probability of the best path, with `vᴱ = τ * max{vᴹ(n,m), vˣ(n, m), vʸ(n, m)}`. This means that the probability is the maximum of the probability of the three states, multiplied by the termination probability `τ`.
*   **Traceback:** Traceback is performed as usual, by using the pointers, to reconstruct the entire alignment.
* **Graphical Representation** A representation of the HMM showing the states and the termination states.

**53. The Viterbi – FSA Connection**

*   **Relationship:** This slide highlights the relationship between Pair HMMs and Finite State Automata (FSA). They are obviously related, however the goal here is to try to formalize what exactly is the relationship.
*   **Precise Connection:** The question is whether there is a specific substitution matrix and affine gap costs such that the FSA alignment is *identical* to the Viterbi path in a pair HMM.
*   **Visual Representation:**  The corresponding representations are shown for both the FSA, and the Pair HMM.

**54. The Viterbi – FSA Connection**

*   **Theorem:** The most probable path from the pair HMM is equal to the optimal alignment using a particular substitution matrix and gap costs.
*   **Substitution Matrix** The substitution matrix is `s(xᵢ, yⱼ) = log(p(xᵢ, yⱼ) / qₓᵢ q_yⱼ) + log((1 - 2δ - τ) / (1-η)²)`. Here, `p(xᵢ, yⱼ)` is the emission probability, `qₓᵢ` and `q_yⱼ` represent gap penalties. The other parameters (δ, τ, η) correspond to the transition probabilities in the model.
*   **Affine Gap Penalty** The affine gap penalty is defined as `γ(g) = -d - (g-1)e`, where `g` is the gap length, `d` is the gap open penalty and `e` is the gap extension penalty. Here, `d = -log(δ(1-ε-τ)/((1-η)(1-2δ-τ)))` and `e=-log(ε/(1-η) )`.
*   **Key Point:** The theorem shows that a pair HMM is a probabilistic generalization of traditional pairwise alignment algorithms (such as gap affine).
*   **Proof:** The proof of this theorem can be found in the exercises.

**55. A Random Model Written as a Pair HMM**

*   **Random Model:** A "random model" for two sequences (which should have no correlation between them), is represented as a pair HMM with two different states `RX` and `RY`, each generating a sequence independently of each other.
*   **No Match State:** This random model does not have a match state.
*   **Independently:** The states `RX` and `RY` emit the two sequences independently from each other.
*  **Visual Representation** A simple model showing how transitions happen between a Begin state, a state `RX` which emits the sequence *x*,  state `RY` which emits sequence *y*, then a silent transitional state, and finally an End state.

**56. A Pair HMM for Local Alignment**

*  **Local alignment model** To represent local alignment, a pair HMM can be built by adding a random model before and after a core alignment (match) model.
*   **Global Model with Flanks** A new model is created by combining a "global model" with the previous random model.
*   **Start and Stop Anywhere:**  With this, you can start and stop the alignment at any location of the sequence, resulting in local alignments.
*   **Unaligned Flanks:** The parts of the sequences that flank the aligned regions are unaligned, and those regions are generated by the random model.
*   **Visual Representation:** The visualization has a copy of the random model at the beginning (`RX₁`, `RY₁`), and at the end `RX₂`, `RY₂`. These models flank a core model with match and insertion states.

**57. The Full Probability of Two Aligned Sequences**

*   **Weak Similarity:** If two sequences are very different, it becomes difficult to find the correct alignment.
*   **HMMs:**  HMMs let us compute the probability that two sequences are related, given *any* alignment `P(X,Y) = ∑_II P(X,Y,II)`.
*   **Viterbi vs. Total:** The total probability of a sequences `P(X,Y)` is always *higher* than the probability of the Viterbi path  `P(X,Y,II*)`. And it can be *significantly different* if there are many comparable alternative alignments.
*   **Better Score:** A better score would be to use the likelihood that two sequences are related by any alignment (match), as opposed to them being unrelated (random): `score(X,Y) = P(X,Y|Match) / P(X,Y|Random)`.  This can be computed as the sum of all the paths `∑_II P(X,Y,II)` divided by the product of the emission probabilities of both sequence `q_x q_y`.
*  **Forward Algorithm:**   You can use the *forward algorithm* to compute `∑_II P(X,Y,II)`.

**58. Recall: The Forward Algorithm**

*   **Review:**  A summary of the forward algorithm, which was already explained in this lecture.
*  **Forward variable:** `fₗ(i)` represents the probability of emitting a prefix of the sequence and reaching the state `πᵢ = l`.
* **Equation:** The values are computed with the equation `fₗ(i) = eₗ(xᵢ) * ∑ₖ fₖ(i-1)aₖₗ`.
*  **Application:** Here, we need to adapt the forward algorithm to pair HMMs to calculate the total probability of two aligned sequences.

**59. The Full Probability: Forward Algorithm**

*  **Pair HMMs:** This slide defines how the forward algorithm is used for Pair HMMs.
*  **Forward variables:** To compute total probability in a pair HMM, we use the variables `fᴹ(i,j)`, `fˣ(i,j)`, and `fʸ(i,j)`.
*   **Equations:**
    *  `fᴹ(i,j) = pₓᵢyⱼ * [(1 - 2δ - τ)fᴹ(i-1, j-1) + (1 - ε - τ)fˣ(i-1, j-1) + (1 - ε - τ)fʸ(i-1, j-1)]` The probability at a match state `M` is the sum over all paths from state M, X and Y in the previous position, times the emission probability `p_xiyi`.
    *  `fˣ(i,j) = qₓᵢ * [δfᴹ(i-1, j) + εfˣ(i-1, j)]`. The probability at an insert to x state, is computed from transitions from M, and other X states, times the emission `q_xᵢ`.
    *    `fʸ(i,j) = q_yⱼ * [δfᴹ(i, j-1) + εfʸ(i, j-1)]`. The probability at an insert to y state, is computed from transitions from M, and other Y states, times the emission `q_yⱼ`.
*  **Diagram:** The visual representation shows the different states in the Pair HMM, and how they transition between each other.

**60. The Full Probability (cont'd)**

*  **Total Probability:** The total probability of the two sequences, according to the forward algorithm is `P(X,Y) = fᴱ = τ [fᴹ(n,m) + fˣ(n,m) + fʸ(n,m)]`.
*  **Posterior over Alignments:** The probability is needed to compute the posterior distribution over all possible alignments of `x` and `y`. The posterior of an alignment `II` is given by `P(II | X,Y) = P(X,Y, II) / P(X,Y)`.
*   **Viterbi Path:** If you are only interested in the Viterbi path, you can replace `II` with the most probable alignment `II*`.
*   **Key Point:** This formula lets us estimate the probability that a given alignment is correct given the sequences we are comparing.

**61. The Full Probability (cont'd - Example)**

*   **Globin Example:** Two globin sequences (HBA_HUMAN and LGB2_LUPLU) are aligned. The standard Viterbi path is computed using a Pair HMM.
*   **Posterior Probability:** The posterior probability of the best (Viterbi) path is very low `P(II* | X,Y) = 4.6 * 10⁻⁶`. This means that the probability that the optimal scoring alignment is correct is extremely low.
*   **Explanation:** The low probability results from the fact that there are many alternative alignments with very similar scores, which means that it is hard to choose just one.

**62. The Posterior Probability**

*   **Conservation:** The degree of conservation in a multiple alignment can vary. Some parts of an alignment are more conserved than others due to functional and structural constraints. Some regions are clear while others are less certain.
*   **Local Accuracy:** Given a multiple alignment, we are usually interested in finding the *local accuracy* of it.
*   **Reliability Measure:**  We need a *reliability measure for each part of the alignment*. This measure tells us how probable is that two characters, at positions `i` and `j` (`xᵢ, yⱼ`)  are aligned, given the complete sequences `(X,Y)`. `P(xᵢ◊yⱼ|X, Y)`.
*   **Backward Algorithm:** To compute this posterior probability you need the *backward algorithm*.

**63. The Backward Algorithm**

*   **Quantity of interest** The main goal is to compute the posterior probability of an alignment at a specific location: `P(xᵢ ◊ yⱼ | X,Y)`. This is the probability that the characters *i* and *j* are aligned given all the data.
*   **Formula:** The formula to calculate the posterior probability is given by `P(xᵢ ◊ yⱼ | X,Y) = P(xᵢ ◊ yⱼ ,X,Y) / P(X, Y)`.
*   **Denominator:** The denominator `P(X,Y)` is the total probability of the alignment given by the *forward algorithm*, which is equal to `fᴱ (n,m)`.
*   **Numerator:** The numerator, `P(X, Y, xᵢ ◊ yⱼ)` is the probability of the sequences and the alignment at the positions *i* and *j*. This is done by combining the forward variable up to position (i,j), `fᴹ(i, j)` and the backward probability of the sequence from (i+1 and j+1), `bᴹ(i, j)`.
*   **Key points:** The backward algorithm calculates the posterior probability of two sequences given an alignment, which gives an indication of the reliability of this alignment.

**64. Recall: The Backward Algorithm**

*   **Review:** This slide reviews the basic structure of the backward algorithm.
*  **Backward variable:** `bₖ(i)` is the probability of emitting the *suffix* `(xᵢ₊₁,...,xₗ)` given that the HMM is at state `πᵢ = k`.
* **Recurrence equation** This variable is calculated recursively, using all the next states and positions. The formula is `bₖ(i) = ∑ₗₑQ aₖₗ eₗ(xᵢ₊₁) bₗ(i+1)`.
* **Use in pair HMM:** The `b(i)` backward variable, can also be used to compute the probability of the suffix alignment in a pair HMM. The probability of the suffix starting at *i+1* and *j+1* is called `b^(M/X/Y)(i,j)`.

**65. The Backward Algorithm: Recursion**

*   **Backward Equations for Pair HMM:** The values of `b^(M/X/Y)(i,j)` are computed recursively.
*   **Recurrence Equations:**
    *    `bᴹ(i, j) = (1 – 2δ – τ)pₓᵢ₊₁yⱼ₊₁ bᴹ(i+1, j+1) + δqₓᵢ₊₁bˣ(i+1,j) + δq_yⱼ₊₁bʸ(i, j+1)` The backward value for state M, depends on a transition to the match state, to an x insertion state, or to an y insertion state.
    *   `bˣ(i, j) = (1 - ε - τ)pₓᵢ₊₁yⱼ₊₁bᴹ(i+1, j+1) + εqₓᵢ₊₁bˣ(i+1, j)`. The value for state X is computed from state M and state X itself, times the emission probability `pₓᵢ₊₁yⱼ₊₁` and `qₓᵢ₊₁`
    *   `bʸ(i, j) = (1 - ε - τ)pₓᵢ₊₁yⱼ₊₁ bᴹ(i+1, j+1) + εq_yⱼ₊₁bʸ(i, j+1)`. Similarly, the value for the state `Y` is computed from transitions to M and Y.
*   **Visual Representation:** The corresponding pair HMM is shown, where the different transitions probabilities and emissions are represented.

**66. Hidden Markov Models: Profile HMMs**

*  **New Topic:** Now we discuss Profile HMMs.
*   **Motivation:** HMMs can be used to compute the probability of one sequence given a profile, instead of just the alignment.
*   **Pairwise vs. Profile Alignments:** You can see the difference in the representation:
    *   **Pairwise alignment:** Two sequences `x` and `y` are aligned.
    *   **Profile Alignment:**  Several sequences are aligned against a profile, with an emission probability for every state.

**67. Profile HMMs**

*   **Sequence Families:** Highly similar sequences are often grouped into *sequence families*.
*   **Query Sequence:**  The goal is to determine whether a query sequence belongs to an existing family.
*   **Multiple Alignment:** We assume that we have a multiple alignment of the sequences that we use to build the profile.
*   **Structural Information:** These multiple alignments are usually built from structural information.
*   **Visual Representation:** A multiple sequence alignment of globin proteins is shown, with conserved and variable regions highlighted with different colors. You can see that different positions have a different conservation profile.
*  **Gaps align up:** The gaps tend to align up between the sequences, meaning that there are blocks of ungapped regions.

**68. Profile HMMs**

*   **Ungapped Blocks:** The first step in building a profile HMM is to consider only the ungapped regions.
*  **Definition:**  A *profile* `P` is defined as the set of probabilities `eᵢ(b)`, which represent the probability of observing the letter `b` at position `i`, which is a column in the multiple alignment.
*   **New Sequence Probability** The probability of a new sequence `X` according to the profile is given by `P(X|P) = ∏ᵢ₌₁L eᵢ(xᵢ)`, where L is the length of the ungapped block.
*   **Visual Representation:** A representation of a profile alignment, showing different values for the probability of seeing a character in the different positions (`eᵢ(xᵢ)`)

**69. Position Specific Score Matrices**

*   **Typical Question:** Is a new sequence a member of the family of sequences from which the profile was built?
*   **Alignment:** The new sequence has to be aligned to the profile.
*   **Likelihood Score:** The membership is tested by evaluating the *likelihood score* of the alignment:
     `score(X|P) = ∑ᵢ₌₁L log(eᵢ(xᵢ)/qₓᵢ)`
*   **Position-Specific Scores:** The value `log(eᵢ(xᵢ)/qₓᵢ)` behaves similarly to the typical scores in alignment `s(a, b)`, where the index `b` is a *position* `i`, instead of an aminoacid. This is why these score matrices are called  *position specific score matrix (PSSM)*.

**70. Position Specific Score Matrices (cont'd)**

*  **PSSM as a trivial HMM:** A PSSM is a very basic HMM where the states (`M₁, ..., Mₗ`) are connected in series.
*  **Start and End:** You can see an initial Begin state, then all the Match states, and finally the End state. The transitions have a value of 1.
* **Emissions:** Each match state emits one character with an emission probability given by `e_Mj(b)`.
* **No choice of transitions** The alignment in this basic PSSM is trivial, there is no choice of state to take.
* **Information loss:**  A simple PSSM is not enough, because gaps in an alignment contain information that is not being captured.
*   **Insertion States:** To model gaps, we include *insertion states* `I₁,...,Iₗ`. We assume that the emission probability from an insertion state is a random probability `q_a`, as done in the pair HMMs.

**71. Position Specific Score Matrices (cont'd)**

*   **Transitions with Insertions:**  The HMM is extended to allow transitions from a Match state to an Insertion state `Mⱼ → Iⱼ`. Transitions between insertion states (`Iⱼ → Iⱼ`), represent gap extensions. Finally transitions can also happen from the insertion state to the next match state `Iⱼ → Mⱼ₊₁`.
*   **Log-Odds Cost of Insert:** The score (or log-odds cost) of insertions can be seen as a sum of the logarithms of the transition probabilities and the emission probabilities.  `log(a_MⱼIⱼ) + log(a_IⱼMⱼ₊₁) + (k-1)log(a_IⱼIⱼ) + log(e_Iⱼ(x)/qₓ)`.
*  **Graphical Representation** You can see how the state Mj is connected to the corresponding insertion state Ij, and how it transitions to the next match state Mj₊₁.

**72. PSSMs: Allowing Deletions**

*   **Deletions:**  Allow for segments of the multiple alignment that are not matched by any character in the query sequence. These are also called "forward jump transitions".
*   **Problem:**  Too many transitions between states are necessary, if you allow all the possible jump transitions.
*   **Solution:** Instead of using many transitions you can introduce *deletion states*, `D₁, ..., Dₗ`.
*   **Silent States:** Deletion states cannot emit any symbols. They are therefore silent states.

**73. Deletions (cont'd)**

*   **Deletions and Parameters:** With the deletion states you drastically reduced the number of parameters required.
*   **High vs Low transition probabilities:** If you allowed forward jump transitions, you could have specific transitions with high probability and others with low probability, which you cannot encode using silent deletion states.
*   **Visual representation:** An example shows that a fully connected model can allow high probabilities for transitions such as `1 -> 5` or `2->4` and low probabilities for `1->4` and `2->5`.  This is not possible with silent states, where you are not specifying individual transition probabilities between different distant states.

**74. A Full Profile HMM Model**

*  **Structure:** A graphical representation of a full profile HMM, including match states (`M`), insertion states (`I`), and deletion states (`D`). Also, the beginning and ending states, where the alignment starts and ends.
*   **Emission Profile:** Here, we are able to align a query sequence to a profile, which is represented with different columns of aligned characters and probabilities.

**75. Profile HMMs are Generalized Pair HMMs**

*   **Special Case:** If a multiple alignment consists of only *one* sequence (i.e. the query sequence), the resulting profile HMM has a particular structure which is equivalent to an “unrolled” version of the pair HMM.
*  **Visual Representation**  The representation shows a series of match states (M1, M2, M3 and M4) where the query sequence is emitted. Also, the model contains insertion states (I), and deletion states (D).
*  **Key Idea:**   A Profile HMM is simply a pair HMM, conditioned to produce a given sequence.

**76. Recall: General Structure of Viterbi Algorithm**

*  **Review** A review of the Viterbi algorithm. Here we will focus on the equations to perform Viterbi in a profile HMM.
*   **Viterbi Variable:**  `vₗ(i)` is the probability of the most probable path for a *prefix* ending in state `l`.
*  **Viterbi equation:** The Viterbi equation is computed as `vₗ(i) = eₗ(xᵢ) max{vₖ(i-1)aₖₗ}`
*  **Log odds:** It is common to use the log-odds scores to avoid underflows: `Vₗ(i) = log(eₗ(xᵢ) / q_xᵢ) + maxₖ{Vₖ(i-1) + log(aₖₗ)}`.  This equation calculates a log-likelihood of the best path for a prefix that ends in a state `πᵢ = l`.

**77. Profile HMM: Viterbi Algorithm**

*   **Match state formula:** The main equations to compute the Viterbi scores for profile HMMs are presented. The Viterbi equation for match states is
 `Vⱼᴹ(i) = log(e_Mⱼ(xᵢ)/qₓᵢ) + max{ Vⱼ₋₁ᴹ(i-1) + log(a_Mⱼ₋₁,Mⱼ) ,  Vⱼ₋₁ᴵ(i-1) + log(a_Iⱼ₋₁,Mⱼ) , Vⱼ₋₁ᴰ(i-1) + log(a_Dⱼ₋₁,Mⱼ) }`
*  **Predecessors:** This equation considers the previous match state, the previous insertion state and the previous deletion state for all locations *j*.
*  **Graphical representation** A visual representation of how the match state `Mj` is connected to the previous states  `Mj₋₁`, `Ij₋₁` and `Dj₋₁`.

**78. Profile HMM: Viterbi Algorithm (cont'd)**

*   **Insertion State Formula:** This slide shows the Viterbi equation for calculating the score of the insertion states. The recurrence is computed using the formula: `Vⱼᴵ(i) = log(e_Iⱼ(xᵢ)/qₓᵢ) + max {Vⱼᴹ(i-1) + log(a_MⱼIⱼ) , Vⱼᴵ(i-1) + log(a_IⱼIⱼ) , Vⱼᴰ(i-1) + log(a_DⱼIⱼ) }`.
     * The previous formula computes the score at insertion states, by maximizing the value of its three predecessors: match, insert and delete.
*   **Emission Probability for Insertions:**  Note that if  `e_Ij(xᵢ)` equals `q_xᵢ`, then the value of `log(e_Iⱼ(xᵢ)/qₓᵢ)` is equal to zero, so there will be no emission scores at that stage.
*   **Graphical Representation** Here we can see that the insertions state `Iⱼ` is connected to the previous states: the previous match state `Mⱼ`, the previous insertion state `Iⱼ` itself, and the previous deletion state `Dⱼ`.

**79. Profile HMM: Viterbi Algorithm (cont'd)**

*   **Deletion State Formula:** This slide provides the Viterbi recurrence equation for the deletion states. They have similar formulas, however because deletion states do not emit any symbols the emission probability has a value of zero.
   * The formula is  `Vⱼᴰ(i) = max{ Vⱼ₋₁ᴹ(i) + log(a_Mⱼ₋₁,Dⱼ) , Vⱼ₋₁ᴵ(i) + log(a_Iⱼ₋₁,Dⱼ) ,  Vⱼ₋₁ᴰ(i) + log(a_Dⱼ₋₁,Dⱼ) }`.
*   **Silent States:** Remember that deletion states are silent states, meaning that they do not emit any characters, so they have no emission probabilities.
*  **Graphical Representation** The visual representation shows how deletion state `Dⱼ` is connected to the previous states `Mj₋₁`, `Iⱼ₋₁` and `Dj₋₁`.

**80. Profile HMM: Viterbi Algorithm (cont'd)**

*   **Complexity:** This slide discusses the complexity of the Viterbi algorithm in the case of profile HMMs.
*   **Number of calculations:**  The total number of cells to compute is `O(L*m)`, where `L` is the length of the sequence, and `m` is the length of the profile.
*   **Operations per Cell:** Calculating each cell takes a constant number of operations `O(1)` because only the 3 predecessors must be considered.
*  **Total Complexity:** Therefore the algorithm runs in `O(L*m)` time, and it also requires `O(L*m)` space.

**81. Profile HMM: Local Alignment**

*   **Local Alignment with Profile HMMs:** To perform local alignment with a profile HMM, the original begin and end states have to be changed.
*   **Silent Transitional States:** The begin and end states are replaced by silent transitional states that are connected to *all* the match states, as well as to themselves.
*  **Random models** To model unaligned portions of the sequences, the profile HMMs are flanked by "random" HMM models.
*  **Key Point** This approach enables the alignment to start and stop anywhere in the sequence.

**82. Application Example**

*  **Globin Data:** A profile HMM was trained with 300 globin protein sequences, and then used to find other globin sequences in the Swiss-Prot database.
*  **Log-likelihood plot** The left plot shows the length-normalized log-likelihood score of every sequence plotted against the protein length.
*   **Log-odds plot** The right plot shows the log-odds score of every sequence plotted against the protein length.
 * **Interpretation**  The results clearly show that the globin sequences obtain a high score, while non-globins are easily distinguished with a low score.

**83. Profile HMM Software**

*   **HMMER:** This is a free and commonly used software package for sequence analysis, written by Sean Eddy.
*   **General Usage:**  It is used to identify homologous (evolutionary related) sequences, both proteins and nucleotides, and perform sequence alignments.
*   **Homology:** HMMER detects homology by comparing a profile HMM against a single sequence or against a database of sequences.
*   **Core Utility:** It is a core utility in protein family databases such as Pfam and InterPro.
*   **Other tools:** Other bioinformatics tools such as UGENE, also uses HMMER.
*   **jackHMMER:** A tool that iteratively searches a database against a protein database, with iterative rounds of refinement. This tool is a core component of the Alphafold package.

That concludes this very long explanation of Hidden Markov Models and their applications. I hope that it is helpful. Please let me know if you need any further clarification or would like to explore some aspects further.
