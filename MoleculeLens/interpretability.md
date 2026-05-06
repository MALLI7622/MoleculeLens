# MoleculeLens Interpretability Research Plan

This document lays out four concrete mechanistic interpretability research directions
that extend the MoleculeLens paper. Each track is self-contained and can be implemented
independently, but together they form a coherent narrative justifying both the
"MoleculeLens" name and the "Explainable" claim in the title.

---

## Why This Architecture Is Unusually Interpretable

Before describing the tracks, it is worth noting what makes MoleculeLens better suited
for closed-form linear-saliency analysis than most contrastive models:

1. **Both projection heads are linear** (`W_m: 2048→256`, `W_t: 768→256`). There is no
   nonlinearity in the bridge, so the text-to-ECFP4 saliency map is available in closed
   form. In the final paper we still treat this as a **linear saliency approximation**
   because the implemented model includes bias terms and an `L2` normalization step that
   are omitted from the simple formula.

2. **ECFP4 bits have ground-truth semantics.** Each bit corresponds to a specific circular
   molecular substructure (radius-2 Morgan fingerprint) that RDKit can decode back to
   atoms and bonds. Features are *already labeled* before any sparse autoencoder is trained.

3. **The 256-d joint space is small.** SAE training on 2,699 × 2 embeddings (drug + text)
   is computationally instant and will not overfit.

4. **A natural counterfactual already exists.** The `--remove_drug_name` ablation condition
   (already run in `03_train_scaffold_split.py`) provides a ground-truth leakage signal
   against which every interpretability analysis can be validated.

**Relevant prior work driving these ideas:**
- Attribution graphs and CLTs: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Sparse crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
- Scaling monosemanticity with SAEs: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Logit lens for LLMs: https://arxiv.org/html/2503.11667v1

---

## Track 1 — ECFP4 Bit Attribution ("Molecular Lens")

**Priority: Highest. No new training required.**

### Core Idea

For any drug-text retrieval pair, compute a closed-form linear saliency score for every ECFP4 bit
to the similarity score, then decode those bits back to RDKit substructures and
highlight them on the molecule.

### Mathematical Formulation

The similarity between drug `i` and text `j` in the shared space is:

```
s_ij = b_i^(m) · b_j^(t)
     = normalize(W_m · x_i) · normalize(W_t · z_j)
```

Because `W_m` is linear, the contribution of ECFP4 bit `k` to the similarity with
text `j` is analytically:

```
attribution_k(i, j) = [W_m^T · b_j^(t)]_k
```

where `b_j^(t) = normalize(W_t · z_j)` is the unit-normalized text embedding.
No sampling, no approximation — the formula is closed-form.

Positive `attribution_k` means bit `k` (a specific molecular substructure) pushes the
drug toward the text in the shared space. Negative means it pushes away.

### Implementation Outline

**Inputs (already saved):**
- `outputs/proj_mol.pt` — W_m (2048 → 256)
- `outputs/proj_text.pt` — W_t (768 → 256)
- `outputs/Bt_test.pt` — normalized text embeddings, shape [435, 256]
- `outputs/Bm_test.pt` — normalized drug embeddings, shape [435, 256]
- `outputs/test_df.csv` — SMILES and metadata for 435 test drugs

**Algorithm:**

```python
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

W_m = torch.load("outputs/proj_mol.pt")          # [256, 2048]
W_t = torch.load("outputs/proj_text.pt")          # [256, 768]
Bt  = torch.load("outputs/Bt_test.pt")            # [435, 256]
Bm  = torch.load("outputs/Bm_test.pt")            # [435, 256]

# Attribution of ECFP4 bits for pair (i, j)
def ecfp_attribution(i, j):
    # attribution vector in ECFP4 space: shape [2048]
    return (W_m.T @ Bt[j])   # W_m^T · b_j^(t)

# Decode top ECFP4 bits back to atoms via RDKit
def highlight_attributions(smiles, attribution_vec, top_k=10):
    mol = Chem.MolFromSmiles(smiles)
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
    top_bits = attribution_vec.abs().topk(top_k).indices.tolist()
    atom_highlights = set()
    for bit in top_bits:
        if bit in info:
            for atom_idx, radius in info[bit]:
                # collect all atoms within 'radius' hops
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                atom_highlights.update(
                    mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in env
                )
                atom_highlights.update(
                    mol.GetBondWithIdx(b).GetEndAtomIdx() for b in env
                )
    return mol, sorted(atom_highlights)
```

**New script to create:** `08_ecfp4_attribution.py`

### Analyses to Run

1. **Per-pair case studies**: Pick 6-8 correctly retrieved test pairs spanning different
   target families (kinase, GPCR, ion channel, nuclear receptor). For each, show the
   molecule with highlighted atoms + the attribution bar chart of top-20 ECFP4 bits.
   This becomes Figure 3 in the paper.

2. **Aggregate attribution across all 435 test pairs**: Which ECFP4 bits appear in the
   top-10 most often? Decode these to substructures. Are they pharmacophore features
   (aromatic rings, H-bond donors) or scaffold-specific? This tests whether the model
   learned generalizable chemistry or scaffold memorization.

3. **Leakage attribution comparison**: For the same drug, compute attribution under
   `text_rich` vs `text_nodrug` projections (load `outputs/proj_text_nodrug.pt`).
   Do the same ECFP4 bits score highly in both conditions? If attribution patterns
   *change* when the drug name is removed, the model was routing signal through name-
   correlated molecular features rather than genuine substructure chemistry.

4. **Wrong-but-close retrieval analysis**: For test pairs where the model retrieved
   rank-2 or rank-3 instead of the true match, compute attribution for both the
   correct and retrieved drug. Do they share high-attribution substructures? If yes,
   the error is chemically reasonable (similar pharmacophore, different scaffold).

### Paper Claim Enabled

> "MoleculeLens provides closed-form, training-free linear saliency attribution for any retrieval decision to
> molecular substructures, enabled by the linear projection design. For any drug-text
> pair, one can identify which circular substructures (radius-2 ECFP4 environments)
> drive alignment, and visualize them directly on the molecular graph."

---

## Track 2 — SAE on the 256-d Shared Space ("Alignment Lens")

**Priority: High. Requires ~1 day of training (trivially fast on CPU).**

### Core Idea

Train a sparse autoencoder (SAE) on the joint 256-d embedding space — both drug
embeddings and text embeddings, all living in the same space after projection. This
reveals whether the alignment dimensions are monosemantic (each dimension = one
pharmacological concept) or polysemantic (each dimension = a mixture).

Directly inspired by Anthropic's "Scaling Monosemanticity" (2024).

### Architecture

```
Encoder: z = ReLU(W_enc(b - b_pre) + b_enc)    # 256 → n_features
Decoder: b̂ = W_dec · z + b_pre                  # n_features → 256
Loss:    L = ||b - b̂||²_2 + λ ||z||_1
```

Train on all 5,398 embeddings = 2,699 drug embeddings + 2,699 text embeddings.
Recommended sizes: n_features ∈ {512, 1024, 2048}. Start with 1024.

Following Anthropic's scaling law findings, reconstruction MSE follows a power law
`L(C) ∝ C^(-α)` with SAE size — train all three sizes and pick the elbow.

### Implementation Outline

**New script:** `09_sae_alignment_space.py`

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=256, n_features=1024):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Linear(d_model, n_features, bias=True)
        self.W_dec = nn.Linear(n_features, d_model, bias=False)
        # Normalize decoder columns to unit norm (standard SAE constraint)
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(
                self.W_dec.weight.data, dim=0
            )

    def forward(self, x):
        x_centered = x - self.b_pre
        z = torch.relu(self.W_enc(x_centered))
        x_hat = self.W_dec(z) + self.b_pre
        return x_hat, z

    def loss(self, x, x_hat, z, lam=1e-3):
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        sparsity = z.abs().sum(dim=-1).mean()
        return recon + lam * sparsity, recon, sparsity
```

Training: Adam, lr=1e-3, 2000 epochs, batch size=256.
Re-normalize decoder columns to unit norm after each gradient step.

**Inputs:**
- `outputs/Bm_test.pt` + encode all training drug embeddings (re-run projection heads
  on training ECFP4s)
- `outputs/Bt_test.pt` + encode all training text embeddings
- `outputs/Bm_test_nodrug.pt`, `outputs/Bt_test_nodrug.pt` for leakage analysis

### Analyses to Run

1. **Feature labeling by drug class**: For each SAE feature `f`, find the top-20
   embeddings (drug or text) with highest `z_f`. Check whether they cluster by target
   family (kinase, GPCR, ion channel, nuclear receptor, protease) or action type
   (agonist, antagonist, inhibitor, activator). Report % features that are
   interpretably labeled by any single category. This is the monosemanticity score.

2. **Cross-modal feature overlap for matched pairs**: For each matched drug-text pair
   `(i, i)`, compute the Jaccard similarity of active features (z > 0.1 threshold):
   ```
   overlap_i = |active(z_drug_i) ∩ active(z_text_i)| / |active(z_drug_i) ∪ active(z_text_i)|
   ```
   High overlap = the model uses the *same* feature basis for drug structure and
   mechanism text. This is the strongest evidence of genuine cross-modal grounding.
   Report mean overlap and compare across target families.

3. **Leakage feature identification**: Train SAE on `text_rich` embeddings. Train
   separately (or use the same SAE) on `text_nodrug` embeddings. Features whose
   activation collapses when drug name is removed are **lexical leakage features**.
   Features that survive are **mechanism features**. Report: how many of the 1024
   features are leakage vs mechanism? What do they correspond to?

4. **Polysemanticity measurement**: For each SAE feature, compute the entropy of its
   activation distribution over target families. Low entropy = monosemantic (activates
   for one family). High entropy = polysemantic. Plot the distribution of entropy
   scores across all features.

5. **Behavioral steering (optional but striking)**: Clamp a specific SAE feature to
   a high value in a drug embedding, recompute the similarity ranking, and show that
   the retrieved mechanism texts shift toward the expected pharmacological category.
   This directly mirrors Anthropic's "Golden Gate Claude" feature clamping experiments.

### Paper Claim Enabled

> "Sparse autoencoder analysis of the 256-dimensional alignment space reveals that
> approximately X% of learned features are monosemantic — each corresponding to a
> single pharmacological concept (kinase inhibition, GPCR antagonism, etc.).
> Leakage features, which collapse under drug-name removal, are identifiable and
> separable from mechanism features that survive the ablation."

---

## Track 3 — Contrastive Logit Lens ("Layer Lens")

**Priority: High. Novel technique. Requires hooking into the frozen RoBERTa.**

### Core Idea

In language models, the logit lens applies the final unembedding matrix `W_u` at
each intermediate layer `ℓ` to track how token predictions evolve across depth.

Here, the analogue is: apply the trained `W_t` projection head at each layer of the
frozen S-Biomed-RoBERTa to ask: "at what layer does the text representation become
*drug-retrievable*?"

This is a genuine novel contribution — logit lens has not been applied to cross-modal
contrastive retrieval settings before.

### Mathematical Formulation

At RoBERTa layer `ℓ`, extract the CLS hidden state `h_ℓ ∈ R^768`. Project it:

```
b_ℓ = normalize(W_t · h_ℓ)    ∈ R^256
```

Then compute full-gallery Recall@1 using `b_ℓ` as the text query against all drug
embeddings `B^(m)`. Plot `Recall@1_ℓ` vs `ℓ` for `ℓ = 0, 1, ..., 12`.

Do this for both `text_rich` and `text_nodrug` conditions. The divergence between
the two curves locates the layer where drug-name information enters.

### Implementation Outline

**New script:** `10_contrastive_logit_lens.py`

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta = AutoModel.from_pretrained(model_name, output_hidden_states=True)
roberta.eval()

W_t = torch.load("outputs/proj_text.pt")  # [256, 768], trained projection head
Bm  = torch.load("outputs/Bm_test.pt")   # [435, 256], drug embeddings

def encode_at_each_layer(texts, batch_size=64):
    """Returns hidden states at all 13 positions (embedding + 12 layers)
    for each text, shape [N, 13, 768]."""
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i:i+batch_size], return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            out = roberta(**batch, output_hidden_states=True)
        # out.hidden_states: tuple of 13 tensors, each [B, seq, 768]
        # take CLS token (index 0)
        cls_states = torch.stack(
            [h[:, 0, :] for h in out.hidden_states], dim=1
        )  # [B, 13, 768]
        all_hidden.append(cls_states)
    return torch.cat(all_hidden, dim=0)  # [N, 13, 768]

def recall_at_1(query_embeddings, gallery_embeddings):
    sims = query_embeddings @ gallery_embeddings.T  # [N, N]
    ranks = sims.argsort(dim=1, descending=True)
    correct = (ranks[:, 0] == torch.arange(len(query_embeddings))).float()
    return correct.mean().item()

# Run for both text conditions
for condition in ["rich", "nodrug"]:
    texts = load_test_texts(condition)  # text_rich or text_nodrug from test_df.csv
    H = encode_at_each_layer(texts)     # [435, 13, 768]
    recall_curve = []
    for layer in range(13):
        h_l = H[:, layer, :]                           # [435, 768]
        b_l = torch.nn.functional.normalize(h_l @ W_t.T, dim=-1)  # [435, 256]
        r1 = recall_at_1(b_l, Bm)
        recall_curve.append(r1)
    # Plot recall_curve for this condition
```

### Analyses to Run

1. **Layer-emergence curve**: Plot R@1 vs RoBERTa layer for both `text_rich` and
   `text_nodrug`. Expected result: `text_rich` curve rises earlier and peaks higher.
   The layer where `text_rich` first significantly outperforms `text_nodrug` is the
   layer where drug-name leakage enters.

2. **Per-target-family layer analysis**: Segment the 435 test examples by target family.
   For kinase inhibitors (largest group), does retrieval emerge at a different layer
   than for GPCRs? Classes with more stereotyped mechanism text may be retrievable
   from earlier layers.

3. **Correct vs incorrect retrieval**: For the pairs where MoleculeLens retrieves
   rank-1 correctly, at which layer does the similarity order "lock in"? Compare to
   pairs where the correct answer is retrieved at rank-2 or rank-3. Late lock-in may
   indicate harder generalization cases.

4. **Token-level attention during layer emergence**: At the identified emergence layer,
   extract the attention weights for the CLS token. Which tokens (mechanism words,
   target names, action type words, drug name tokens) does CLS attend to most at the
   layer where retrieval ability appears? This localizes leakage to specific attention
   heads.

### Paper Claim Enabled

> "A contrastive logit lens analysis reveals that drug-retrievable information in the
> frozen S-Biomed-RoBERTa encoder emerges predominantly in layers 8-12, while drug-
> name leakage is detectable as early as layer 4. This provides layer-level mechanistic
> evidence for the leakage effect quantified in the ablation tables."

(The specific layer numbers are hypotheses to be confirmed by the experiment.)

---

## Track 4 — Cross-Modal Crosscoder ("Alignment Feature Discovery")

**Priority: Medium. Extends Track 2. Most novel architectural contribution.**

### Core Idea

Anthropic's sparse crosscoders find features shared *across transformer layers*,
resolving "cross-layer superposition." Here, the analogous problem is *cross-modal
superposition*: a feature relevant to "kinase inhibition" should be active in both
the drug embedding (from ECFP4 substructures) and the text embedding (from mechanism
words). A vanilla SAE trained on one modality cannot see the other.

A cross-modal crosscoder reads both `b^(m)` and `b^(t)` jointly for matched pairs
and learns features shared *across modalities*.

### Architecture

For matched drug-text pair `(b^(m)_i, b^(t)_i)` where both are 256-d:

```
Input:   [b^(m); b^(t)] ∈ R^512        (concatenated pair)
Encode:  z = ReLU(W_enc · input + b_enc)    # 512 → n_features
Decode:  [b̂^(m); b̂^(t)] = W_dec · z + b_pre    # n_features → 512
Loss:    L = ||b^(m) - b̂^(m)||² + ||b^(t) - b̂^(t)||² + λ||z||_1
```

A feature `f` is a **cross-modal alignment feature** if it has large activation
weight for both the drug reconstruction sub-block and the text reconstruction
sub-block (look at `W_dec[0:256, f]` and `W_dec[256:512, f]` norms).

A feature is a **text-only feature** if `||W_dec[256:512, f]||` >> `||W_dec[0:256, f]||`.
These are candidate leakage features.

A feature is a **molecule-only feature** if `||W_dec[0:256, f]||` >> `||W_dec[256:512, f]||`.
These encode structural properties not captured in the text.

### Analyses to Run

1. **Feature-type census**: Report the proportion of cross-modal vs text-only vs
   molecule-only features. A model doing genuine structure-mechanism alignment should
   have more cross-modal features than leakage-only models.

2. **Leakage localization**: Among text-only features, find those whose activation
   correlates with drug name presence (compare activations on `text_rich` vs
   `text_nodrug` pairs). These are the mechanistic origin of the 41% leakage drop.

3. **Cross-modal feature geometry**: For cross-modal features, check whether the
   drug-side direction `W_dec[0:256, f]` aligns with known pharmacophore directions
   in ECFP4 space (using Track 1 attribution). Does the text-side direction
   `W_dec[256:512, f]` align with mechanism-word dimensions in RoBERTa output space?

4. **Feature ablation on retrieval**: Zero out a specific cross-modal feature in both
   modalities for a test pair and recompute similarity. Report the Recall@1 degradation
   from ablating top-10 cross-modal features. This directly mirrors the attribution
   graph perturbation experiments in the Anthropic 2025 paper.

### Paper Claim Enabled

> "A cross-modal crosscoder trained on matched drug-text embedding pairs identifies
> three feature types: cross-modal alignment features (active in both modalities),
> text-only leakage features (active only in text, correlated with drug name presence),
> and molecule-only structural features. Ablating the top cross-modal features reduces
> Recall@1 by X points, providing causal evidence for their role in retrieval."

---

## Summary Table

| Track | Method Inspired By | New Training? | Expected Paper Impact |
|---|---|---|---|
| 1 — ECFP4 Attribution | Attribution graphs (Anthropic 2025) | None | Figure 3: molecule visualizations |
| 2 — SAE on 256-d space | Scaling Monosemanticity (Anthropic 2024) | ~10 min (CPU) | Table 3: feature census + leakage features |
| 3 — Contrastive Logit Lens | LogitLens4LLMs (arXiv 2503.11667) | None | Figure 4: layer-emergence curves |
| 4 — Cross-modal Crosscoder | Sparse Crosscoders (Anthropic 2024) | ~30 min (CPU) | Table 4 + Figure 5: feature-type census |

**Recommended implementation order: Track 1 → Track 3 → Track 2 → Track 4**

Track 1 requires no new code dependencies beyond RDKit (already in the environment).
Track 3 only needs `output_hidden_states=True` added to the existing RoBERTa call.
Tracks 2 and 4 require writing the SAE training loops but are computationally trivial.

---

## Connecting to the Paper Narrative

The four tracks together support a single unified story:

1. **Track 1** justifies the "lens" metaphor literally — you can see inside every
   retrieval decision at the substructure level.

2. **Track 3** shows that the frozen text encoder is not a black box — specific
   RoBERTa layers carry different kinds of information relevant to retrieval,
   and leakage enters at a locatable depth.

3. **Track 2** shows that the 256-d shared space is not arbitrary — it decomposes
   into interpretable pharmacological features, and leakage features are
   separable from mechanism features.

4. **Track 4** provides the strongest causal claim: there exist specific cross-modal
   features whose ablation predictably degrades retrieval, moving the result from
   correlation to mechanism.

These analyses also reframe the 41% leakage result from the main paper. Rather than
just reporting a number, the interpretability tracks show *where* leakage enters
(Track 3), *what form* it takes in the embedding space (Track 2), and *which
molecular features co-vary with it* (Track 1). That is what earns the title
"MoleculeLens: An Explainable Thin Bridge."



Research Plan: Adding Explainability to MoleculeLens                                                                                                                                              
                                                                                                                                                                                                    
  Here's what I think is scientifically compelling, genuinely novel, and feasible with your existing architecture. I'll propose 4 interpretability analyses, graded by impact and implementability. 
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  Why Your Architecture Is Particularly Well-Suited                                                                                                                                                 
                                                                                                                                                                                                    
  Your model has a rare property that makes it better suited for mechanistic interpretability than most contrastive models:                                                                         
                                                                                                                                                                                                    
  1. Both projection heads are linear (W_m: 2048→256, W_t: 768→256). This makes a closed-form linear saliency map available without SHAP or integrated-gradients-style estimators, although the final paper  
  still treats the formula as an approximation because the implementation includes bias terms and `L2` normalization.                                                                                
  2. ECFP4 bits have ground-truth semantics — each bit maps to a specific circular substructure via RDKit. This means features are already labeled before any SAE is trained.                       
  3. The 256-d joint space is tiny — SAE training on 2,699 × 2 embeddings is instant and won't overfit.                                                                                             
  4. You have a natural counterfactual: the --remove_drug_name condition. This lets you test interpretability claims against a known source of leakage.                                             
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  Track 1 — ECFP4 Bit Attribution ("Molecular Lens") ⭐ Highest Impact                                                                                                                              
                                                                                                                                                                                                    
  What it does: For any drug-text retrieval pair, identify which molecular substructures (ECFP4 bits) receive the highest closed-form linear saliency under the matched text.
                                                                                                                                                                                                    
  Why the linear saliency is still useful:                                                                                                                                                          
                                                                                                                                                                                                    
  The similarity is:                                                                                                                                                                                
  s_ij = normalize(W_m · x_i) · normalize(W_t · z_j)
                                                    
  Since W_m is linear, the contribution of ECFP4 bit k to similarity with text j is:                                                                                                                
  attribution_k = [W_m^T · b_j^(t)]_k  where b_j^(t) = normalize(W_t · z_j)                                                                                                                         
                                                                                                                                                                                                    
  This is the dot product of the k-th column of W_m with the projected text embedding. It is analytically computed and cheap, but in the final paper it is described as a linear approximation because     
  the implemented model also includes bias terms and `L2` normalization.                                                                                                                            
                                                                                                                                                                                                    
  What you can show in the paper:                                                                                                                                                                   
  - For a drug correctly retrieved by its "GPCR antagonist" mechanism text, highlight the aromatic ring and amine substructures that drove the match                                                
  - For a drug incorrectly retrieved when drug name is present vs. absent, show which bits change importance — mechanistically tracing the leakage                                                  
  - Aggregate across the 435 test pairs: which ECFP4 substructures are universally important for high-confidence retrieval?                       
                                                                                                                                                                                                    
  Novelty claim: "MoleculeLens provides closed-form, training-free linear saliency attribution of retrieval decisions to molecular substructures, enabled by the linear projection design"             
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  Track 2 — SAE on the 256-d Shared Space ("Alignment Lens") ⭐ Most Aligned with Anthropic Work                                                                                                    
                                                                                                                                                                                                    
  What it does: Train a sparse autoencoder on the joint 256-d embedding space (both drug and text embeddings), revealing whether the alignment dimensions are monosemantic or polysemantic.
                                                                                                                                                                                                    
  Architecture (directly from Scaling Monosemanticity):                                                                                                                                             
  Encoder: z = ReLU(W_enc(b - b_pre) + b_enc)    # 256 → 1024                                                                                                                                       
  Decoder: b̂ = W_dec(z) + b_pre                   # 1024 → 256                                                                                                                                      
  Loss:    L = ||b - b̂||² + λ||z||₁                                                                                                                                                                 
                                                                                                                                                                                                    
  Train on all 5,398 embeddings (2,699 drug + 2,699 text in shared space).                                                                                                                          
                                                                                                                                                                                                    
  Key analyses:                                                                                                                                                                                     
  - Feature labeling: For each of the 1024 SAE features, find which drugs/mechanisms maximally activate it → correlate with target family (kinase, GPCR, ion channel...) or action type (agonist,   
  antagonist, inhibitor...)                                                                                                                                                                         
  - Leakage features: Compare SAE activations on text_rich vs text_nodrug embeddings. Features that collapse when drug name is removed = lexical leakage features. Features that survive = mechanism
   features                                                                                                                                                                                         
  - Cross-modal alignment geometry: For matched drug-text pairs, are the same SAE features active? High overlap = true mechanistic alignment. Low overlap with high similarity = the model is using 
  a different representational route in each modality                                                                                                                                              
                                                                                                                                                                                                    
  Novelty claim: "First SAE analysis of a cross-modal contrastive alignment space; identifies separable 'mechanism features' and 'leakage features' in the shared representation"
                                                                                                                                                                                                    
  ---             
  Track 3 — Contrastive Logit Lens ("Layer Lens") ⭐ Most Novel Technique                                                                                                                           
                                                                                                                                                                                                    
  What it does: Apply your trained W_t projection at each layer of frozen RoBERTa to ask: "at what layer does the text representation become drug-retrievable?"
                                                                                                                                                                                                    
  The analogy to Logit Lens: In language models, logit lens applies the final unembedding matrix W_u at layer ℓ to get p_ℓ(next token | context). Here, you apply W_t at each RoBERTa layer ℓ and   
  compute retrieval performance:                                                                                                                                                                    
                                                                                                                                                                                                    
  for ℓ in range(12):  # RoBERTa layers
      h_ℓ = roberta_hidden_states[ℓ][:, 0, :]   # CLS at layer ℓ, shape [N, 768]                                                                                                                    
      b_ℓ = normalize(W_t @ h_ℓ.T).T             # project to 256-d, normalize                                                                                                                      
      recall_at_1[ℓ] = compute_recall(b_ℓ, B_m_test)                                                                                                                                                
                                                                                                                                                                                                    
  This produces a layer-vs-retrieval-performance curve.                                                                                                                                             
                                                                                                                                                                                                    
  What it reveals:                                                                                                                                                                                  
  - Which RoBERTa layers carry drug-structure-relevant information? (usually later layers)
  - When you remove drug names: does the retrieval-emergence layer shift or does it just decrease? A shift in emergence layer would strongly suggest drug names are shortcutted in earlier layers   
  - Does the "with drug name" curve have an earlier inflection point than "no drug name"? That would be direct mechanistic evidence of where in the text encoder the leakage enters              
                                                                                                                                                                                                    
  Novelty claim: "Novel adaptation of logit lens to cross-modal contrastive retrieval; reveals that drug-name leakage enters the representation at RoBERTa layers 4-8, while mechanism information  
  emerges in layers 9-12" (hypothesis — to be verified)                                                                                                                                             
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  Track 4 — Cross-Modal Crosscoder ("Alignment Feature Discovery")
                                                                                                                                                                                                    
  What it does: Inspired by crosscoders (which find features shared across transformer layers), train a joint SAE that reads both b^(m) and b^(t) and learns features shared across modalities.
                                                                                                                                                                                                    
  Architecture:   
  Input: [b^(m); b^(t)] ∈ R^512  (concatenated drug + text embedding for matched pairs)                                                                                                             
  SAE:   z = ReLU(W_enc · input + b_enc)   # 512 → 1024                                
         [b̂^(m); b̂^(t)] = W_dec · z + b_pre  # reconstruct both                                                                                                                                     
                                                                                                                                                                                                    
  Features with high activation in both the drug and text sub-reconstruction are cross-modal alignment features. Features active only in the text reconstruction are text-only features (candidates 
  for leakage).                                                                                                                                                                                     
                                                                                                                                                                                                    
  This is less critical than Tracks 1-3 but completes the story.                                                                                                                                    
                  
  ---                                                                                                                                                                                               
  How These Connect to Your Existing Claims
                                                                                                                                                                                                    
  ┌────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                      Paper Claim                       │                                           Interpretability Evidence                                           │                        
  ├────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ "Thin bridge remains competitive under scaffold split" │ Track 1: closed-form substructure saliency shows the model is using structural features, not memorized scaffolds │                    
  ├────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ "Drug-name leakage is real but 41% vs 80.5%"           │ Track 2: SAE identifies leakage features; Track 3: layer lens shows where leakage enters                      │                        
  ├────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                        
  │ "MoleculeLens" naming is justified                     │ Track 1: literal visual "lens" into which molecular substructures drive each retrieval decision               │                        
  ├────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                        
  │ "Frozen encoders + linear projections are competitive" │ Tracks 1-2: linearity is necessary for closed-form saliency analysis — end-to-end models lose this                │
  └────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                        
                  
  ---                                                                                                                                                                                               
  Suggested Implementation Order
                                
  1. Track 1 — 1-2 days of coding, produces the most visually compelling paper figure (highlighted molecule diagrams)
  2. Track 3 — 1 day, hooks on frozen RoBERTa, requires saving layer-wise CLS states                                                                                                                
  3. Track 2 — 2-3 days, SAE training loop + feature validation + leakage analysis                                                                                                                  
  4. Track 4 — Optional, extends Track 2                                                                                                                                                            
                                                                                                                                                                                                    
  ---                                                                                                                                                                                               
  Should I start implementing any of these? I'd suggest beginning with Track 1 since it requires no new training (just linear algebra on your already-saved proj_mol.pt, proj_text.pt, and          
  Bt_test.pt, Bm_test.pt) and will produce the clearest visualization for the paper. Track 3 requires hooking into the frozen RoBERTa forward pass, which I can set up after.                       
   
