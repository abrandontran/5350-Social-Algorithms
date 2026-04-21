"""
Final Project: Utility-Constrained Causal Serendipity Recommender System 
(S&DS 5350, Social Algorithms)

Author: Cailey Bobadilla
Dataset: Last.fm HetRec 2011
(CITATION)

Pipeline:
- Stage 1: Exposure Modeling
    - Poisson Factorization (PF) on binary exposure matrix A to estimate the
      natural probability that each user discovers each artist (a_hat_ui).
- Stage 2: Outcome Model
    - Causal Logistic Matrix Factorization (LMF) with the substitute confounder
      (a_hat_ui) injected into the logistic prediction. The user-specific
      coefficient gamma_u absorbs the exposure bias so that the remaining
      latent vectors theta_u and beta_i represent pure, unconfounded preference.
- Stage 3: Utility Optimization
    - Empirical estimation of per-category satisfaction rate q_t, followed by
      Peng et al.'s inverse-weight slot allocation to maximize util_1.
- Stage 4: Slate Assembly
    - Causal serendipity scoring (S_ui = preference / exposure) and top-N
      slate construction subject to category quotas.

AI Acknowledgement:
- Claude Sonnet 4.6
- Gemini 3.1 Pro
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')
from hpfrec import HPF


# -----------------------------------------------------------------------------
# CONFIGURATION AND HYPERPARAMETERS
# -----------------------------------------------------------------------------

# Path to unzipped dataset folder
DATA_DIR = './data/hetrec2011-lastfm-2k'

# LMF confidence scaling factor (alpha): c_ui = alpha * r_ui
# NOTE: 
# - Converts raw listen counts into a statistical confidence measure
# - Higher values amplify the signal from heavy listeners
# - Value of 40 follows the original LMF paper (Johnson, 2014)
ALPHA = 40

# Confidence threshold for a satisfying interaction (tau_c)
# NOTE: 
# - A user-artist pair is considered a "satisfying" interaction if c_ui >= TAU_C
# - Used to compute empirical category satisfaction rates (q_t)
# - TAU_C = 400 corresponds to ~5 raw listens * ALPHA=40
TAU_C = 200

# Minimum unconfounded preference score (tau_r)
# NOTE:
# - Items where theta_u @ beta_i < TAU_R are dropped from the candidate pool
#   before slate assembly
# - Acts as a quality floor to prevent recommending items the model has low 
#   confidence the user will enjoy, even if their exposure probability is near 
#   zero (which would otherwise inflate S_ui)
TAU_R = -1.0

# Small constant to prevent division by zero in the serendipity ratio S_ui
EPSILON = 1e-8

# Total number of recommendation slots in the final user-facing slate
N_SLOTS = 10

# Number of latent dimensions for both the Poisson Factorization and LMF models
N_FACTORS = 20

# Number of alternating gradient ascent iterations for training the LMF
N_ITER = 50

# Number of top crowdsourced tags to retain as distinct item categories
# NOTE: All other tags are collapsed into an "other" category to prevent 
#       excessive fragmentation of the category space
TOP_K_TAGS = 10


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data(data_dir: str):
    """
    Loads and zero-indexes the three relevant Last.fm HetRec 2011 files.

    File roles:
        user_artists.dat --> r_ui (raw listening count, our implicit feedback 
                                   signal)
        user_taggedartists.dat --> artist-to-category mapping (via tag merging)
        tags.dat --> human-readable tag labels
        artists.dat --> artist metadata for readable output

    Args:
        data_dir (str): Path to the unzipped hetrec2011-lastfm-2k folder.

    Returns:
        ua (pd.DataFrame): User-artist interactions with columns
                           [userID, artistID, weight, u, i], where u and i
                           are zero-indexed integers for matrix indexing.
        artists (pd.DataFrame): Artist metadata with columns
                                [artistID, name, url, pictureURL].
        tags (pd.DataFrame): Tag vocabulary with columns [tagID, tagValue].
        uta (pd.DataFrame): User-artist-tag assignments with columns
                            [userID, artistID, tagID, day, month, year,
                            tagValue].
        user2idx (dict): Maps original userID --> zero-indexed integer.
        item2idx (dict): Maps original artistID --> zero-indexed integer.
        n_users (int): Total number of unique users.
        n_items (int): Total number of unique items (artists).
    """
    # user_artists.dat: columns are userID, artistID, weight
    # NOTE: 'weight' is the raw listen count r_ui — the core implicit feedback 
    #       signal
    ua = pd.read_csv(
        f'{data_dir}/user_artists.dat',
        sep='\t',
        names=['userID', 'artistID', 'weight'],
        header=0
    )

    # artists.dat: columns are id, name, url, pictureURL
    # NOTE: Used only for mapping item indices back to readable artist names in 
    #       output
    artists = pd.read_csv(
        f'{data_dir}/artists.dat',
        sep='\t',
        names=['artistID', 'name', 'url', 'pictureURL'],
        header=0,
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # tags.dat: columns are tagID, tagValue
    # NOTE: Provides human-readable labels for crowdsourced tags
    tags = pd.read_csv(
        f'{data_dir}/tags.dat',
        sep='\t',
        names=['tagID', 'tagValue'],
        header=0,
        encoding='latin-1'
    )

    # user_taggedartists.dat: columns are userID, artistID, tagID, day, month,
    # year
    # NOTE: Used to assign each artist to a macro-category via plurality tag
    uta = pd.read_csv(
        f'{data_dir}/user_taggedartists.dat',
        sep='\t',
        names=['userID', 'artistID', 'tagID', 'day', 'month', 'year'],
        header=0
    )

    # Merge tag labels into uta so we have tagValue strings alongside tagIDs
    uta = uta.merge(tags, on='tagID', how='left')

    # Build zero-indexed mappings for matrix factorization compatibility
    # NOTE: Sorting ensures deterministic ordering across runs
    users = sorted(ua['userID'].unique())
    items = sorted(ua['artistID'].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    # Add integer index columns to ua for direct matrix construction
    ua['u'] = ua['userID'].map(user2idx)
    ua['i'] = ua['artistID'].map(item2idx)

    n_users = len(users)
    n_items = len(items)

    print(f'Loaded: {n_users} users, {n_items} artists, {len(ua)} interactions')

    return ua, artists, tags, uta, user2idx, item2idx, n_users, n_items


# -----------------------------------------------------------------------------
# MATRIX CONSTRUCTION
# -----------------------------------------------------------------------------

def build_matrices(ua: pd.DataFrame, n_users: int, n_items: int, mode='linear',
                   alpha=ALPHA):
    """
    Constructs the three foundational sparse matrices for the pipeline from
    the user-artist interaction dataframe.

    Args:
        ua (pd.DataFrame): User-artist interaction dataframe. Must contain
                           columns 'u' (user index), 'i' (item index), and
                           'weight' (raw listen count r_ui).
        n_users (int): Number of unique users.
        n_items (int): Number of unique items (artists).
        mode (str): Linear or logarithmic scaling for confidence.
        alpha (float): LMF confidence scaling factor. Defaults to ALPHA.

    Returns:
        R (csr_matrix): Raw interaction matrix of shape (n_users, n_items).
                        R[u, i] = r_ui (raw listen count).
        C (csr_matrix): Confidence matrix of shape (n_users, n_items).
                        C[u, i] = alpha * r_ui = c_ui.
        A (csr_matrix): Binary exposure matrix of shape (n_users, n_items).
                        A[u, i] = 1 if any interaction exists, 0 otherwise.
                        This is the treatment assignment matrix a_ui.
    """
    rows = ua['u'].values
    cols = ua['i'].values
    weights = ua['weight'].values

    # R: raw listen counts (r_ui)
    R = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items))

    # C: confidence weighted counts (construction based on mode)
    if mode == 'linear':
        # c_ui = alpha * r_ui
        C_data = alpha * weights
    elif mode == 'log':
        # c_ui = 1 + alpha * log(1 + r_ui)
        C_data = 1 + alpha * np.log1p(weights)

    # C: confidence-weighted counts (c_ui = alpha * r_ui)
    C = csr_matrix((C_data, (rows, cols)), shape=(n_users, n_items))

    # A: binary exposure (a_ui = 1 if r_ui > 0, else 0)
    # NOTE: np.ones fills every observed entry with 1.0 regardless of listen 
    #       count
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

    print(f'Sparsity: {100 * (1 - R.nnz / (n_users * n_items)):.2f}%')

    return R, C, A


# -----------------------------------------------------------------------------
# STAGE 1: EXPOSURE MODELING (POISSON FACTORIZATION)
# -----------------------------------------------------------------------------

def fit_poisson_factorization(A: csr_matrix, n_factors: int = N_FACTORS):
    """
    Models the MCAR violation by fitting Hierarchical Poisson Factorization
    (HPF) to the binary exposure matrix A.

    HPF decomposes A into user activity factors (Theta) and item popularity
    factors (Beta), capturing the confounding mechanisms (e.g., genre
    preferences, mainstream popularity) that drive organic discovery. The
    resulting a_hat_ui estimates the probability that user u would naturally
    encounter artist i without algorithmic intervention.

    Args:
        A (csr_matrix): Binary exposure matrix of shape (n_users, n_items).
                        A[u, i] = 1 if the user has ever listened to the artist.
        n_factors (int): Number of latent dimensions for the HPF model.
                         Defaults to N_FACTORS.

    Returns:
        model (HPF): Fitted HPF model object. model.Theta is the user latent 
                     matrix (n_users x k); model.Beta is the item latent matrix 
                     (n_items x k).
        a_hat (np.ndarray): Dense array of shape (n_users, n_items) containing
                            exposure probabilities in [0, 1].
    """
    model = HPF(k=n_factors, verbose=True, random_seed=42)

    # hpfrec expects a DataFrame with columns: UserId, ItemId, Count
    # NOTE: Extract nonzero positions from the sparse matrix for efficiency
    df_exposure = pd.DataFrame({
        'UserId': A.nonzero()[0],
        'ItemId': A.nonzero()[1],
        # Flatten to 1D: A is binary so all nonzero values are 1.0
        'Count': np.array(A[A.nonzero()]).flatten()
    })

    model.fit(df_exposure)

    # Reconstruct full dense exposure score matrix via dot product of factors:
    # raw_score[u, i] = Theta[u] @ Beta[i]  (shape: n_users x n_items)
    a_hat = model.Theta @ model.Beta.T

    # Min-max normalize to [0, 1] to preserve relative differences in exposure
    a_min, a_max = a_hat.min(), a_hat.max()
    a_hat = (a_hat - a_min) / (a_max - a_min + EPSILON)

    print(f'a_hat range: [{a_hat.min():.4f}, {a_hat.max():.4f}]')
    print(f'a_hat std: {a_hat.std():.4f}')

    return model, a_hat


# -----------------------------------------------------------------------------
# STAGE 2: OUTCOME MODEL (CAUSAL LMF)
# -----------------------------------------------------------------------------

class CausalLMF:
    """
    Logistic Matrix Factorization (LMF) modified for causal debiasing.

    Standard LMF (Johnson 2014) models implicit feedback as:
        p(l_ui) = sigmoid(theta_u @ beta_i)

    This class injects Wang et al.'s substitute confounder (a_hat_ui) directly
    into the prediction, yielding:
        p(l_ui) = sigmoid(theta_u @ beta_i + gamma_u * a_hat_ui)

    During training, gamma_u absorbs the exposure bias correlated with a_hat.
    At inference time, gamma_u is dropped (see predict_unconfounded), leaving
    theta_u @ beta_i as the pure, unconfounded preference estimate.

    Training uses full-batch alternating gradient ascent with confidence
    weighting. All updates are vectorized across users or items simultaneously.
    """

    def __init__(self, n_users, n_items, n_factors=N_FACTORS, n_iter=N_ITER,
                 lr=0.01, reg=0.01, random_state=42, init_Theta=None, 
                 init_Beta=None):
        """
        Initializes the CausalLMF model parameters.

        Args:
            n_users (int): Number of users in the dataset.
            n_items (int): Number of items (artists) in the dataset.
            n_factors (int): Number of latent dimensions. Defaults to N_FACTORS.
            n_iter (int): Number of alternating gradient ascent iterations. 
                          Defaults to N_ITER.
            lr (float): Learning rate for all parameter updates. Defaults to 
                        0.01.
            reg (float): L2 regularization coefficient applied to Theta, Beta, 
                         and Gamma to prevent overfitting. Defaults to 0.01.
            random_state (int): Seed for reproducibility. Defaults to 42.
            init_Theta (np.ndarray or None): Optional pre-trained user latent
                                             matrix of shape 
                                             (n_users, n_factors) for a warm
                                             start from PF factors. Defaults to 
                                             None.
            init_Beta (np.ndarray or None): Optional pre-trained item latent
                                            matrix of shape (n_items, n_factors) 
                                            for a warm start from PF factors. 
                                            Defaults to None.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.lr = lr
        self.reg = reg
        rng = np.random.default_rng(random_state)
        scale = 0.01
        self.history = []

        # Warm-start from PF latent factors if provided
        # NOTE: Leverages the structural information already captured by the 
        #       exposure model, which can improve convergence speed and final 
        #       performance
        if init_Theta is not None and init_Beta is not None:
            print('  Initializing with PF latent factors (Warm Start)')
            self.Theta = init_Theta.copy()   # shape: (n_users, n_factors)
            self.Beta = init_Beta.copy()     # shape: (n_items, n_factors)
        else:
            self.Theta = rng.normal(0, scale, (n_users, n_factors))
            self.Beta = rng.normal(0, scale, (n_items, n_factors))

        # Gamma is always randomly initialized — it is a bias-absorption term
        # specific to the causal outcome model, not present in PF
        self.Gamma = rng.normal(0, scale, (n_users,))  # shape: (n_users,)

    @staticmethod
    def sigmoid(x):
        """
        Numerically stable sigmoid function.

        Uses the piecewise form to avoid overflow in np.exp for large |x|
        values, and clips inputs to [-500, 500] as an extra safeguard.

        Args:
            x (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Element-wise sigmoid of x, in the range (0, 1).
        """
        x = np.clip(x, -500, 500)
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def fit(self, C: csr_matrix, a_hat: np.ndarray):
        """
        Trains the causal LMF via full-batch alternating gradient ascent.

        Each iteration performs two vectorized update steps:
            1. Fix Beta; compute gradients and update Theta and Gamma together
               (both depend on the same error matrix).
            2. Recompute scores with updated Theta/Gamma; fix Theta and Gamma;
               update Beta.

        The confidence matrix C down-weights unobserved pairs (c_ui = 0)
        relative to observed ones (c_ui = alpha * r_ui), preventing the large
        number of zeros from dominating the gradient.

        Args:
            C (csr_matrix): Sparse confidence matrix of shape (n_users,
                            n_items). C[u, i] = alpha * r_ui.
            a_hat (np.ndarray): Dense exposure probability matrix of shape
                                (n_users, n_items) from Stage 1.

        Returns:
            self: The fitted CausalLMF instance.
        """
        # Convert to dense for vectorized ops
        # NOTE: For very large datasets (e.g. >50K items), subsample or switch
        # to a mini-batch approach to avoid memory issues
        C_dense = np.array(C.todense())

        # Binary preference indicator: P[u, i] = 1 if any interaction, else 0
        # NOTE: Serves as the "label" in the logistic objective
        P = (C_dense > 0).astype(float)

        for iteration in range(self.n_iter):
            # Step 1: Update Theta and Gamma (Beta fixed)
 
            # Compute prediction scores for all (u, i) pairs simultaneously
            # NOTE: Gamma[:, np.newaxis] broadcasts (n_users,) to 
            #       (n_users, n_items) so that each user's gamma_u scales their 
            #       entire row of a_hat
            scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
            probs = self.sigmoid(scores)

            # Confidence-weighted residuals: positive for under-predicted pairs,
            # negative for over-predicted pairs
            # NOTE: C_dense down-weights zeros.
            errors = C_dense * (P - probs)   # shape: (n_users, n_items)

            # Gradient for Theta: each user's gradient is the sum of weighted
            # item vectors for all items, minus L2 regularization
            grad_theta = errors @ self.Beta - self.reg * self.Theta
            self.Theta += self.lr * grad_theta

            # Gradient for Gamma: scalar per user, equal to the dot product of
            # the error vector with that user's exposure probabilities
            grad_gamma = np.sum(errors * a_hat, axis=1) - self.reg * self.Gamma
            self.Gamma += self.lr * grad_gamma

            # Step 2: Update Beta (Theta and Gamma fixed) 

            # Recompute scores with the just-updated Theta and Gamma
            scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
            probs = self.sigmoid(scores)
            errors = C_dense * (P - probs)

            # Gradient for Beta: errors.T @ Theta gives each item's gradient
            # as the sum of weighted user vectors across all users
            grad_beta = errors.T @ self.Theta - self.reg * self.Beta
            self.Beta += self.lr * grad_beta

            # Monitor convergence via confidence-weighted log-likelihood
            if (iteration + 1) % 5 == 0:
                scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
                probs = self.sigmoid(scores)
                ll = np.sum(C_dense * (P * np.log(probs + 1e-10) + 
                                       (1 - P) * np.log(1 - probs + 1e-10)))
                self.history.append(ll)

                if (iteration + 1) % 5 == 0:
                    print(f'  Iter {iteration+1}/{self.n_iter} | '
                          f'Log-likelihood: {ll:.2f}')

        return self

    def predict_unconfounded(self, u: int) -> np.ndarray:
        """
        Returns the unconfounded preference scores for all items for user u.

        Gamma is intentionally excluded here. This is the core causal mechanism:
        by dropping gamma_u * a_hat_ui at inference time, we isolate the pure
        latent preference theta_u @ beta_i, which is no longer confounded by
        the exposure mechanism.

        Args:
            u (int): Zero-indexed target user index.

        Returns:
            np.ndarray: 1D array of shape (n_items,) with unconfounded
                        preference scores for every item.
        """
        # Dot product of user u's latent vector with all item vectors
        return self.Theta[u] @ self.Beta.T


# -----------------------------------------------------------------------------
# STAGE 3: UTILITY OPTIMIZATION
# -----------------------------------------------------------------------------

def build_artist_categories(uta: pd.DataFrame, item2idx: dict,
                             top_k: int = TOP_K_TAGS) -> dict:
    """
    Assigns each artist to a macro-category based on its plurality tag —
    the single most frequently applied crowdsourced tag across all users.

    Only the top_k most globally common tags are retained as distinct
    categories. All other tags are collapsed into 'other' to limit the
    category space and avoid over-fragmentation in slot allocation.

    Args:
        uta (pd.DataFrame): User-tagged artists dataframe with at least columns
                            [artistID, tagValue].
        item2idx (dict): Maps original artistID --> zero-indexed integer.
        top_k (int): Number of top tags to keep as named categories. Defaults to 
                     TOP_K_TAGS.

    Returns:
        item_to_category (dict): Maps item index (int) --> category string.
                                 Every item in item2idx is represented;
                                 untagged artists map to 'other'.
    """
    # Count how many times each (artist, tag) pair appears across all users
    tag_counts = uta.groupby(['artistID', 'tagValue']).size().reset_index(name='count')

    # Assign each artist its single most-used tag (plurality tag)
    primary_tag = (tag_counts
                   .sort_values('count', ascending=False)
                   .drop_duplicates('artistID')  # keep only the top tag per artist
                   .set_index('artistID')['tagValue'])

    # Identify the globally top_k most common tags to use as category names
    top_tags = set(uta['tagValue'].value_counts().head(top_k).index)

    item_to_category = {}
    for artist_id, idx in item2idx.items():
        tag = primary_tag.get(artist_id, 'other')
        # Collapse any tag not in the top_k set into 'other'
        item_to_category[idx] = tag if tag in top_tags else 'other'

    categories = list(set(item_to_category.values()))
    print(f'Categories ({len(categories)}): {categories}')
    return item_to_category


def compute_qt(C: csr_matrix, A: csr_matrix,
               item_to_category: dict, tau_c: float = TAU_C) -> dict:
    """
    Empirically estimates the per-category conditional satisfaction rate q_t.

    Following the algorithm's formulation:
        q_t = sum_{u, i in t} I(c_ui >= tau_c)  /  sum_{u, i in t} a_ui

    This is the proportion of exposures within category t that resulted in a
    high-confidence (satisfying) interaction, aggregated across all users.

    Args:
        C (csr_matrix): Confidence matrix of shape (n_users, n_items).
        A (csr_matrix): Binary exposure matrix of shape (n_users, n_items).
        item_to_category (dict): Maps item index --> category string.
        tau_c (float): Confidence threshold for a satisfying interaction. 
                       Defaults to TAU_C.

    Returns:
        qt (dict): Maps category string --> float q_t in [0, 1].
                   Categories with zero exposures receive a floor of 1e-6 to
                   prevent division by zero in downstream slot allocation.
    """
    # Convert to dense for column-slice operations by category
    C_arr = np.array(C.todense())
    A_arr = np.array(A.todense())

    categories = set(item_to_category.values())
    qt = {}

    for cat in categories:
        # Collect all item indices belonging to this category
        cat_items = [i for i, c in item_to_category.items() if c == cat]
        if not cat_items:
            continue

        # Slice the full matrix to only the columns for this category
        c_cat = C_arr[:, cat_items]   # shape: (n_users, |cat_items|)
        a_cat = A_arr[:, cat_items]   # shape: (n_users, |cat_items|)

        # Count satisfying interactions and total exposures across all users
        satisfying = np.sum(c_cat >= tau_c)
        exposed = np.sum(a_cat)
        qt[cat] = satisfying / exposed if exposed > 0 else 1e-6

    print('Category satisfaction rates (q_t):')
    # Sort by the q_t value (second element of each dict item)
    for cat, q in sorted(qt.items(), key=lambda x: -x[1]):
        print(f'  {cat:30s}  q_t = {q:.4f}')

    return qt


def allocate_slots(qt: dict, n_slots: int = N_SLOTS) -> dict:
    """
    Implements Peng et al.'s 'Milk and Ice Cream' corollary via inverse-weight
    slot allocation to maximize util_1 across the slate.

    Categories with lower q_t receive more slots (inverse weighting), because
    more exposures are needed to achieve at least one satisfying interaction.
    This directly operationalizes the corollary that serendipitous (low q_t)
    categories require disproportionately more slots.

    Slot allocation formula:
        w_t = 1 / q_t
        n_t = round(N * w_t / sum(w_k))

    Rounding correction: any integer remainder is added to or subtracted from
    the category with the largest fractional remainder (i.e., the category
    whose rounded allocation diverges most from its exact share).

    Args:
        qt (dict): Maps category string --> q_t float.
        n_slots (int): Total number of recommendation slots N. Defaults to 
                       N_SLOTS.

    Returns:
        n_t (dict): Maps category string --> integer number of allocated slots.
                    Every category receives at least 1 slot (enforced by max).
    """
    categories = list(qt.keys())

    # Inverse weights: lower q_t → higher weight → more slots
    w = {cat: 1.0 / (qt[cat] + EPSILON) for cat in categories}
    total_w = sum(w.values())

    # Exact (floating-point) allocation before rounding
    raw_alloc = {cat: n_slots * w[cat] / total_w for cat in categories}

    # Round to integers, guaranteeing at least 1 slot per category
    n_t = {cat: max(1, round(v)) for cat, v in raw_alloc.items()}

    # Compute how many slots we're over or under due to rounding
    diff = n_slots - sum(n_t.values())
    if diff != 0:
        # Adjust the category whose rounded value diverged most from its exact
        # allocation (largest fractional remainder), as it's the fairest target
        remainders = {cat: abs(raw_alloc[cat] - round(raw_alloc[cat]))
                      for cat in categories}
        adj_cat = max(remainders, key=remainders.get)
        n_t[adj_cat] = max(1, n_t[adj_cat] + diff)

    print(f'\nSlot allocation (N={n_slots}):')
    # Sort by the integer slot count (second element)
    for cat, n in sorted(n_t.items(), key=lambda x: -x[1]):
        print(f'  {cat:30s}  n_t = {n}')

    return n_t


# -----------------------------------------------------------------------------
# STAGE 4: SERENDIPITY SCORING + SLATE ASSEMBLY
# -----------------------------------------------------------------------------

def compute_serendipity_scores(u: int, model: CausalLMF, a_hat: np.ndarray, 
                               A: csr_matrix, tau_r: float = TAU_R, 
                               epsilon: float = EPSILON) -> np.ndarray:
    """
    Computes the causal serendipity score S_ui for user u across all items.

    The serendipity score is defined as:
        S_ui = (theta_u @ beta_i) / (a_hat_ui + epsilon)

    A high S_ui indicates an item the user is likely to enjoy (high numerator)
    but unlikely to discover organically (low denominator). Observed items and
    items below the relevance threshold tau_r are masked with -inf so they are
    never selected during slate assembly.

    Args:
        u (int): Zero-indexed target user.
        model (CausalLMF): Trained causal LMF model.
        a_hat (np.ndarray): Dense exposure probability matrix of shape
                            (n_users, n_items).
        A (csr_matrix): Binary exposure matrix for masking observed items.
        tau_r (float): Minimum unconfounded preference score. Items with
                       theta_u @ beta_i < tau_r are excluded. Defaults to TAU_R.
        epsilon (float): Small constant added to a_hat_ui to prevent
                         zero-division. Defaults to EPSILON.

    Returns:
        S (np.ndarray): 1D array of shape (n_items,) with serendipity scores.
                        Observed and irrelevant items are set to -np.inf.
    """
    # Unconfounded preference scores for all items (Gamma dropped at inference)
    pref = model.predict_unconfounded(u)   # shape: (n_items,)

    # Identify observed items (a_ui = 1) and mask them out — we only recommend
    # items the user has not yet encountered
    observed_mask = np.array(A[u].todense()).flatten() > 0
    pref[observed_mask] = -np.inf

    # Mask items that fall below the minimum relevance threshold tau_r
    # NOTE: Prevents items with near-zero exposure (which would inflate S_ui)
    #       from being recommended if the model also has low preference for them
    pref[pref < tau_r] = -np.inf

    # Compute serendipity score: relevance penalized by natural discoverability
    S = pref / (a_hat[u] + epsilon)

    # Re-apply the observed mask to S (pref is already masked, but S may carry
    # residual values from the division; this guarantees clean -inf for all
    # previously seen items)
    S[observed_mask] = -np.inf

    return S


def build_slate(u: int, model: CausalLMF, a_hat: np.ndarray,
                A: csr_matrix, item_to_category: dict,
                n_t: dict, artists: pd.DataFrame, item2idx: dict) -> pd.DataFrame:
    """
    Assembles the final top-N recommendation slate for user u.

    For each category t, selects the top n_t unobserved items ranked by S_ui.
    The final slate is sorted globally by S_ui for readability, with rank 1
    being the most serendipitous item overall.

    Args:
        u (int): Zero-indexed target user.
        model (CausalLMF): Trained causal LMF model.
        a_hat (np.ndarray): Dense exposure probability matrix of shape
                            (n_users, n_items).
        A (csr_matrix): Binary exposure matrix for candidate masking.
        item_to_category (dict): Maps item index --> category string.
        n_t (dict): Maps category string --> integer slot count.
        qt (dict): Per-category satisfaction rates (passed for completeness; not 
                   used directly in assembly).
        artists (pd.DataFrame): Artist metadata for readable name lookup.
        item2idx (dict): Maps original artistID --> zero-indexed index.

    Returns:
        slate_df (pd.DataFrame): Final recommendation slate with columns
                                 [artist, category, S_ui, pref_score,
                                 exposure_prob], indexed from rank 1 to N.
    """
    S = compute_serendipity_scores(u, model, a_hat, A)

    # Reverse mapping: zero-indexed integer --> original artistID
    idx2item = {v: k for k, v in item2idx.items()}

    slate = []
    for cat, slots in n_t.items():
        if slots == 0:
            continue

        # Collect unobserved, above-threshold items in this category
        cat_items = np.array([i for i, c in item_to_category.items()
                               if c == cat and S[i] > -np.inf])
        if len(cat_items) == 0:
            continue

        # Rank this category's candidates by S_ui (descending) and take top n_t
        cat_scores = S[cat_items]
        top_indices = cat_items[np.argsort(cat_scores)[::-1][:slots]]

        for item_idx in top_indices:
            artist_id = idx2item[item_idx]

            # Look up the human-readable artist name from the metadata dataframe
            name_series = artists.loc[artists['artistID'] == artist_id, 'name']
            artist_name = (name_series.values[0]
                           if len(name_series) > 0
                           else f'Artist_{artist_id}')

            slate.append({
                'artist': artist_name,
                'category': cat,
                'S_ui': S[item_idx],
                'pref_score': model.predict_unconfounded(u)[item_idx],
                'exposure_prob': a_hat[u, item_idx],
            })

    # Sort the full slate by S_ui descending and assign readable rank labels
    slate_df = (pd.DataFrame(slate)
                  .sort_values('S_ui', ascending=False)
                  .reset_index(drop=True))
    slate_df.index += 1
    slate_df.index.name = 'rank'

    return slate_df


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    """
    Executes the full four-stage pipeline end-to-end.

    Stages:
        1. Load raw Last.fm data and construct sparse matrices (R, C, A).
        2. Fit Poisson Factorization to A to get exposure probabilities a_hat.
        3. Fit the Causal LMF on C and a_hat to learn unconfounded latent
           vectors Theta and Beta (and bias-absorbing Gamma).
        4. Estimate category satisfaction rates q_t, compute inverse-weight
           slot allocation n_t, and assemble the final recommendation slate.

    Returns:
        lmf (CausalLMF): Trained outcome model.
        a_hat (np.ndarray): Exposure probability matrix from Stage 1.
        slate (pd.DataFrame): Final recommendation slate for the target user.
    """
    print('=' * 60)
    print('Loading data')
    print('=' * 60)
    ua, artists, tags, uta, user2idx, item2idx, n_users, n_items = load_data(DATA_DIR)

    print('\n' + '=' * 60)
    print('Building matrices')
    print('=' * 60)
    R, C, A = build_matrices(ua, n_users, n_items)

    print('\n' + '=' * 60)
    print('Stage 1: Exposure Modeling (Poisson Factorization)')
    print('=' * 60)
    pf_model, a_hat = fit_poisson_factorization(A, n_factors=N_FACTORS)

    # Serialize a_hat so Stage 2+ can reload it without re-running Stage 1
    # NOTE: Poisson Factorization is slow; this avoids redundant retraining when
    #       iterating on the LMF or slate logic
    np.save('a_hat.npy', a_hat)
    print('Saved a_hat.npy')

    print('\n' + '=' * 60)
    print('Stage 2: Outcome Model (Causal LMF)')
    print('=' * 60)

    # Warm-start LMF with PF latent factors to improve convergence.
    # NOTE: For large datasets (many items), consider subsampling to top artists
    #       by interaction count, or switching to mini-batch SGD.
    lmf = CausalLMF(n_users, n_items, n_factors=N_FACTORS, n_iter=N_ITER, 
                    init_Theta=pf_model.Theta, init_Beta=pf_model.Beta)
    lmf.fit(C, a_hat)

    print('\n' + '=' * 60)
    print('Stage 3: Utility Optimization')
    print('=' * 60)
    item_to_category = build_artist_categories(uta, item2idx, top_k=TOP_K_TAGS)
    qt = compute_qt(C, A, item_to_category, tau_c=TAU_C)
    n_t = allocate_slots(qt, n_slots=N_SLOTS)

    print('\n' + '=' * 60)
    print('Stage 4: Slate Assembly (example: user index 0)')
    print('=' * 60)

    # Change target_user to any integer in [0, n_users) to generate a slate for
    # a different user
    target_user = 0
    slate = build_slate(u=target_user, model=lmf, a_hat=a_hat, A=A,
                        item_to_category=item_to_category, n_t=n_t, 
                        artists=artists, item2idx=item2idx)

    print(f'\nRecommendation slate for user index {target_user}:')
    print(slate.to_string())

    return lmf, a_hat, slate


if __name__ == '__main__':
    lmf, a_hat, slate = main()