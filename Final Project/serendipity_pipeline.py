"""
Final Project (S&DS 5350, Social Algorithms):
    Unexpectedness by Design: Causal Serendipity Optimization for 
    Implicit Recommender Systems

Author: Cailey Bobadilla
Dataset: Last.fm HetRec 2011

Pipeline:
- Stage 1: Exposure Modeling
    - Poisson Factorization (PF) on binary exposure matrix A to estimate the
      natural probability that each user discovers each artist (a_hat_ui).
- Stage 2: Outcome Modeling
    - Standard LMF (baseline): p(l_ui) = sigma(theta_u @ beta_i)
    - Causal LMF: p(l_ui) = sigma(theta_u @ beta_i + gamma_u * a_hat_ui)
      The user-specific coefficient gamma_u absorbs the exposure bias so
      that theta_u and beta_i represent pure, unconfounded preference.
- Stage 3: Serendipity Scoring and Slate Assembly
    - Applies a lower-bound exposure floor (winsorization) to prevent the 
      serendipity score from exploding due to near-zero exposure denominators.
    - Standard LMF: build_std_slate()
    - Causal LMF: build_causal_slate()
    - Causal + Serendipity: build_serendipity_slate()

AI Acknowledgement:
    Claude Sonnet 4.6 and Gemini 3.1 Pro were used to help develop the functions
    and classes included in this script.
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

# Minimum unconfounded preference score (tau_r)
# NOTE:
# - Items where theta_u @ beta_i < TAU_R are dropped from the candidate pool
#   before slate assembly
# - Acts as a quality floor to prevent recommending items the model has low 
#   confidence the user will enjoy, even if their exposure probability is near 
#   zero (which would otherwise inflate S_ui)
TAU_R = -10.0

# Small additive constant for the original (non-winsorized) serendipity ratio.
# NOTE: 
# - Prevents a literal ZeroDivisionError but still allows the denominator to 
#   approach zero, which can produce extreme S_ui outliers 
# - Uses winsorization for a statistically robust alternative (see 
#   compute_serendipity_scores() and compute_dynamic_epsilon())
EPSILON = 1e-8

# Default lower-bound floor used when winsorization is enabled
# NOTE:
# - Caps the maximum serendipity multiplier and prevents score explosions from 
#   near-zero exposure estimates
# - Setting this to 0.01 means every artist is treated as having at least a 1%
#   natural discovery probability
# - Override with compute_dynamic_epsilon() for a data-driven threshold
WINSORIZE_EPSILON = 0.01

# Total number of recommendation slots in the final user-facing slate
N_SLOTS = 10

# Number of latent dimensions for both the Poisson Factorization and LMF models
N_FACTORS = 20

# Number of alternating gradient ascent iterations for training the LMF
N_ITER = 50

# Number of top crowdsourced tags to retain as distinct item categories
# NOTE: All other tags are collapsed into an "other" category to prevent 
#   excessive fragmentation of the category space
TOP_K_TAGS = 10



# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple:
    """
    Loads and zero-indexes the four relevant Last.fm HetRec 2011 files.

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
            [userID, artistID, weight, u, i], where u and i are zero-indexed 
            integers for matrix indexing.
        artists (pd.DataFrame): Artist metadata with columns 
            [artistID, name, url, pictureURL].
        tags (pd.DataFrame): Tag vocabulary with columns [tagID, tagValue].
        uta (pd.DataFrame): User-artist-tag assignments with columns
            [userID, artistID, tagID, day, month, year, tagValue].
        user2idx (dict): Maps original userID --> zero-indexed integer.
        item2idx (dict): Maps original artistID --> zero-indexed integer.
        n_users (int): Total number of unique users.
        n_items (int): Total number of unique items (artists).
    """
    # user_artists.dat: r_ui (raw listen count, core implicit feedback signal)
    ua = pd.read_csv(
        f'{data_dir}/user_artists.dat',
        sep='\t',
        names=['userID', 'artistID', 'weight'],
        header=0
    )

    # artists.dat: artist metadata used for readable name lookup in slate output
    artists = pd.read_csv(
        f'{data_dir}/artists.dat',
        sep='\t',
        names=['artistID', 'name', 'url', 'pictureURL'],
        header=0,
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # tags.dat: human-readable tag labels merged into uta
    tags = pd.read_csv(
        f'{data_dir}/tags.dat',
        sep='\t',
        names=['tagID', 'tagValue'],
        header=0,
        encoding='latin-1'
    )

    # user_taggedartists.dat: used to assign each artist to a macro-category via 
    # plurality tag
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

def build_matrices(ua: pd.DataFrame, n_users: int, n_items: int, 
                   mode: str='linear', alpha: float=ALPHA) -> tuple:
    """
    Constructs the three foundational sparse matrices for the pipeline from the 
    user-artist interaction dataframe.

    Args:
        ua (pd.DataFrame): User-artist interaction dataframe. Must contain
            columns 'u' (user index), 'i' (item index), and 'weight' (raw listen 
            count r_ui).
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
            A[u, i] = 1 if any interaction exists, 0 otherwise. This is the 
            treatment assignment matrix a_ui.
    """
    rows = ua['u'].values
    cols = ua['i'].values
    weights = ua['weight'].values

    # Raw listen counts (r_ui)
    R = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items))

    # Confidence weighted counts: formula depends on mode
    if mode == 'linear':
        # c_ui = alpha * r_ui
        C_data = alpha * weights
    elif mode == 'log':
        # c_ui = 1 + alpha * log(1 + r_ui)
        C_data = 1 + alpha * np.log1p(weights)

    # Confidence-weighted counts (formula depends on mode)
    C = csr_matrix((C_data, (rows, cols)), shape=(n_users, n_items))

    # Binary exposure (a_ui = 1 if r_ui > 0, else 0)
    # NOTE: np.ones fills every observed entry with 1.0 regardless of listen 
    #   count
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

    print(f'Sparsity: {100 * (1 - R.nnz / (n_users * n_items)):.2f}%')

    return R, C, A



# -----------------------------------------------------------------------------
# STAGE 1: EXPOSURE MODELING
# -----------------------------------------------------------------------------

def fit_poisson_factorization(A: csr_matrix, 
                              n_factors: int=N_FACTORS) -> tuple:
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
# STAGE 2: OUTCOME MODELING
# -----------------------------------------------------------------------------

class StandardLMF:
    """
    Standard Logistic Matrix Factorization without causal debiasing.

    Mirrors CausalLMF in architecture and training procedure exactly, but
    omits the substitute confounder term (a_hat_ui) and Gamma entirely.
    The prediction is:
        p(l_ui) = sigma(theta_u @ beta_i)

    Because the exposure signal never enters training, the learned latent
    vectors conflate genuine preference with organic discoverability. Used
    as a controlled baseline: the only difference from CausalLMF is the
    absence of the causal injection.

    No item bias is included in either model so that both implementations
    match the theoretical formulations from Johnson (2014) and Wang et al.
    exactly.
    """

    def __init__(self, n_users, n_items, n_factors=N_FACTORS, n_iter=N_ITER,
                 lr=0.01, reg=0.01, random_state=42, init_Theta=None,
                 init_Beta=None) -> None:
        """
        Initializes the StandardLMF model parameters.

        Identical to CausalLMF.__init__ except that no Gamma vector is
        initialized, since the standard model omits the substitute confounder
        entirely. Used as a controlled baseline for comparison.

        Args:
            n_users (int): Number of users in the dataset.
            n_items (int): Number of items (artists) in the dataset.
            n_factors (int): Number of latent dimensions. Defaults to N_FACTORS.
            n_iter (int): Number of alternating gradient ascent iterations.
                Defaults to N_ITER.
            lr (float): Learning rate for all parameter updates. Defaults to 
                0.01.
            reg (float): L2 regularization coefficient applied to Theta and 
                Beta. Defaults to 0.01.
            random_state (int): Seed for reproducibility. Defaults to 42.
            init_Theta (np.ndarray or None): Optional pre-trained user latent
                matrix of shape (n_users, n_factors) for a warm start from PF 
                factors. Defaults to None.
            init_Beta (np.ndarray or None): Optional pre-trained item latent
                matrix of shape (n_items, n_factors) for a warm start from PF 
                factors. Defaults to None.
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

        if init_Theta is not None and init_Beta is not None:
            print('  Initializing with PF latent factors (Warm Start)')
            self.Theta = init_Theta.copy()
            self.Beta  = init_Beta.copy()
        else:
            self.Theta = rng.normal(0, scale, (n_users, n_factors))
            self.Beta  = rng.normal(0, scale, (n_items, n_factors))

    @staticmethod
    def sigmoid(x) -> np.ndarray:
        """
        Numerically stable sigmoid function (identical to Causal LMF).

        Uses the piecewise form to avoid overflow in np.exp for large |x|
        values, and clips inputs to [-500, 500] as an extra safeguard.

        Args:
            x (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Element-wise sigmoid of x, in the range (0, 1).
        """
        x = np.clip(x, -500, 500)
        return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def fit(self, C: csr_matrix) -> 'StandardLMF':
        """
        Trains the standard LMF via full-batch alternating gradient ascent.

        Mirrors CausalLMF.fit() exactly (including C_total baseline confidence 
        and gradient clipping) except that no Gamma term is present and a_hat is 
        never used. This ensures that any difference in learned representations 
        is attributable solely to the causal injection.

        Args:
            C (csr_matrix): Sparse confidence matrix of shape 
                (n_users, n_items). C[u, i] = alpha * r_ui.

        Returns:
            self: The fitted StandardLMF instance.
        """
        C_dense = np.array(C.todense())
        P = (C_dense > 0).astype(float)

        # Baseline confidence for unobserved items (identical to CausalLMF)
        baseline_conf = np.ones_like(C_dense)
        C_total = baseline_conf + C_dense

        CLIP_VAL = 10.0

        for iteration in range(self.n_iter):

            # Step 1: Update Theta (Beta fixed)
            scores = self.Theta @ self.Beta.T  # no gamma * a_hat term
            probs  = self.sigmoid(scores)
            errors = C_total * (P - probs)

            grad_theta = errors @ self.Beta - self.reg * self.Theta
            self.Theta += self.lr * np.clip(grad_theta, -CLIP_VAL, CLIP_VAL)

            # Step 2: Update Beta (Theta fixed)
            scores = self.Theta @ self.Beta.T
            probs  = self.sigmoid(scores)
            errors = C_total * (P - probs)  # consistent C_total here

            grad_beta = errors.T @ self.Theta - self.reg * self.Beta
            self.Beta += self.lr * np.clip(grad_beta, -CLIP_VAL, CLIP_VAL)

            if (iteration + 1) % 5 == 0:
                scores = self.Theta @ self.Beta.T
                probs  = self.sigmoid(scores)
                ll = np.sum(C_total * (P * np.log(probs + 1e-10) + 
                                       (1 - P) * np.log(1 - probs + 1e-10)))
                self.history.append(ll)
                print(f'  Iter {iteration+1}/{self.n_iter} | '
                      f'Log-likelihood: {ll:.2f}')

        return self

    def predict(self, u: int) -> np.ndarray:
        """
        Returns preference scores for all items for user u.

        The scores reflect a conflation of genuine preference and exposure
        bias because the model was never trained with the causal confounder.

        Args:
            u (int): Zero-indexed target user index.

        Returns:
            np.ndarray: 1D array of shape (n_items,) with preference scores.
        """
        return self.Theta[u] @ self.Beta.T


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
                 init_Beta=None) -> None:
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
                matrix of shape (n_users, n_factors) for a warm start from PF 
                factors. Defaults to None.
            init_Beta (np.ndarray or None): Optional pre-trained item latent 
                matrix of shape (n_items, n_factors) for a warm start from PF 
                factors. Defaults to None.
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
        #   exposure model, which can improve convergence speed and final 
        #   performance
        if init_Theta is not None and init_Beta is not None:
            print('  Initializing with PF latent factors (Warm Start)')
            self.Theta = init_Theta.copy()  # shape: (n_users, n_factors)
            self.Beta = init_Beta.copy()    # shape: (n_items, n_factors)
        else:
            self.Theta = rng.normal(0, scale, (n_users, n_factors))
            self.Beta = rng.normal(0, scale, (n_items, n_factors))

        # Gamma is always randomly initialized — it is a bias-absorption term
        # specific to the causal outcome model, not present in PF
        self.Gamma = rng.normal(0, scale, (n_users,))  # shape: (n_users,)

    @staticmethod
    def sigmoid(x) -> np.ndarray:
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

    def fit(self, C: csr_matrix, a_hat: np.ndarray) -> 'CausalLMF':
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
            C (csr_matrix): Sparse confidence matrix of shape 
                (n_users, n_items). C[u, i] = alpha * r_ui.
            a_hat (np.ndarray): Dense exposure probability matrix of shape
                (n_users, n_items) from Stage 1.

        Returns:
            self: The fitted CausalLMF instance.
        """
        # Convert to dense for vectorized ops
        # NOTE: For very large datasets (e.g. >50K items), subsample or switch
        #   to a mini-batch approach to avoid memory issues
        C_dense = np.array(C.todense())

        # Binary preference indicator: P[u, i] = 1 if any interaction, else 0
        # NOTE: Serves as the "label" in the logistic objective
        P = (C_dense > 0).astype(float)

        # Baseline confidence of 1 for all unobserved pairs so the model learns
        # negative signal rather than ignoring zero entries entirely
        baseline_conf = np.ones_like(C_dense)
        C_total = baseline_conf + C_dense

        # Define a clipping threshold to prevent exploding gradients
        CLIP_VAL = 10.0

        for iteration in range(self.n_iter):
            # Step 1: Update Theta and Gamma (Beta fixed)
 
            # Compute prediction scores for all (u, i) pairs simultaneously
            # NOTE: Gamma[:, np.newaxis] broadcasts (n_users,) to 
            #   (n_users, n_items) so that each user's gamma_u scales their 
            #   entire row of a_hat
            scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
            probs = self.sigmoid(scores)

            # Use C_total to ensure unobserved items generate negative gradients
            errors = C_total * (P - probs)

            # Gradient for Theta: each user's gradient is the sum of weighted
            # item vectors for all items, minus L2 regularization
            grad_theta = errors @ self.Beta - self.reg * self.Theta
            # Apply clipping here
            self.Theta += self.lr * np.clip(grad_theta, -CLIP_VAL, CLIP_VAL)

            # Gradient for Gamma: scalar per user, equal to the dot product of
            # the error vector with that user's exposure probabilities
            grad_gamma = np.sum(errors * a_hat, axis=1) - self.reg * self.Gamma
            # Apply clipping here
            self.Gamma += self.lr * np.clip(grad_gamma, -CLIP_VAL, CLIP_VAL)

            # Step 2: Update Beta (Theta and Gamma fixed) 

            # Recompute scores with the just-updated Theta and Gamma
            scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
            probs = self.sigmoid(scores)
            errors = C_total * (P - probs)

            # Gradient for Beta: errors.T @ Theta gives each item's gradient
            # as the sum of weighted user vectors across all users
            grad_beta = errors.T @ self.Theta - self.reg * self.Beta
            # Apply clipping here
            self.Beta += self.lr * np.clip(grad_beta, -CLIP_VAL, CLIP_VAL)

            # Monitor convergence via confidence-weighted log-likelihood
            if (iteration + 1) % 5 == 0:
                scores = self.Theta @ self.Beta.T + self.Gamma[:, np.newaxis] * a_hat
                probs = self.sigmoid(scores)
                ll = np.sum(C_total * (P * np.log(probs + 1e-10) + 
                                       (1 - P) * np.log(1 - probs + 1e-10)))
                self.history.append(ll)

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
# ARTIST CATEGORIZATION
# -----------------------------------------------------------------------------

def build_artist_categories(uta: pd.DataFrame, item2idx: dict, 
                            top_k: int=TOP_K_TAGS) -> dict:
    """
    Assigns each artist to a macro-category based on its plurality tag —
    the single most frequently applied crowdsourced tag across all users.

    Only the top_k most globally common tags are retained as distinct
    categories. All other tags are collapsed into 'other' to keep the category
    space interpretable in slate output.

    Args:
        uta (pd.DataFrame): User-tagged artists dataframe with at least columns
            [artistID, tagValue].
        item2idx (dict): Maps original artistID --> zero-indexed integer.
        top_k (int): Number of top tags to keep as named categories. Defaults to 
            TOP_K_TAGS.

    Returns:
        item_to_category (dict): Maps item index (int) --> category string. 
            Every item in item2idx is represented; untagged artists map to 
            'other'.
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



# -----------------------------------------------------------------------------
# STAGE 3: SERENDIPITY SCORING + SLATE ASSEMBLY
# -----------------------------------------------------------------------------

def compute_dynamic_epsilon(a_hat: np.ndarray, 
                            percentile: float = 5.0) -> float:
    """
    Computes a data-driven winsorization floor as the p-th percentile of all
    strictly positive exposure probabilities in a_hat.

    Rather than assuming a fixed floor (e.g., 0.01), this approach scales the
    threshold automatically to the actual distribution of exposure scores in
    the dataset. Use this once after Stage 1 and pass the result as epsilon
    to compute_serendipity_scores() with winsorize=True.

    Args:
        a_hat (np.ndarray): Dense exposure probability matrix of shape 
            (n_users, n_items) from Stage 1.
        percentile (float): Percentile of non-zero exposures to use as the 
            floor. Defaults to 5.0 (5th percentile).

    Returns:
        float: Data-driven epsilon floor value.
    """
    nonzero_exposures = a_hat[a_hat > 0]
    dynamic_eps = float(np.percentile(nonzero_exposures, percentile))
    print(f'Dynamic epsilon (p{percentile:.0f} of non-zero exposures): '
          f'{dynamic_eps:.6f}')
    return dynamic_eps


def compute_serendipity_scores(u: int, model: CausalLMF, a_hat: np.ndarray, 
                               A: csr_matrix, tau_r: float = TAU_R, 
                               epsilon: float = EPSILON,
                               winsorize: bool = False) -> np.ndarray:
    """
    Computes the causal serendipity score S_ui for user u across all items.

    Two denominator modes are supported:

    Original (additive shift, winsorize=False):
        S_ui = (theta_u @ beta_i) / (a_hat_ui + epsilon)
    Winsorized (lower-bound capping, winsorize=True):
        S_ui = (theta_u @ beta_i) / max(a_hat_ui, epsilon)

    The additive shift only prevents a literal ZeroDivisionError but still
    allows the denominator to approach epsilon from above, producing extreme
    S_ui outliers when epsilon is tiny (e.g., 1e-8). Winsorization rounds any
    exposure below epsilon up to epsilon, enforcing a hard mathematical floor
    on item "invisibility" and capping the maximum serendipity multiplier.
    Pass a larger, statistically meaningful epsilon (e.g., 0.01 or the output
    of compute_dynamic_epsilon()) when winsorize=True.

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
        epsilon (float): Denominator floor value.
            - winsorize=False: added to a_hat_ui (original). Use a tiny value 
              like EPSILON (1e-8).
            - winsorize=True: hard lower bound via np.clip. Use a meaningful 
              threshold like WINSORIZE_EPSILON (0.01) or 
              compute_dynamic_epsilon(a_hat).
        winsorize (bool): If True, apply winsorization (max(a_hat_ui, epsilon))
            instead of the additive shift. Defaults to False to preserve 
            backward compatibility.

    Returns:
        S (np.ndarray): 1D array of shape (n_items,) with serendipity scores.
            Observed and irrelevant items are set to -np.inf.
    """
    # Unconfounded preference scores for all items (Gamma dropped at inference)
    pref = model.predict_unconfounded(u)   # shape: (n_items,)

    # Identify observed items (a_ui = 1) and mask them out (only recommend items 
    # the user has not yet encountered)
    observed_mask = np.array(A[u].todense()).flatten() > 0
    pref[observed_mask] = -np.inf

    # Mask items that fall below the minimum relevance threshold tau_r
    # NOTE: Prevents items with near-zero exposure (which would inflate S_ui)
    #   from being recommended if the model also has low preference for them
    pref[pref < tau_r] = -np.inf

    # Compute denominator based on chosen mode
    if winsorize:
        # WINSORIZATION: hard lower bound — no exposure can be treated as
        # lower than epsilon, regardless of how close to zero a_hat falls.
        # np.clip with a_min=epsilon achieves max(a_hat_ui, epsilon).
        denominator = np.clip(a_hat[u], a_min=epsilon, a_max=None)
    else:
        # ORIGINAL: additive shift — prevents ZeroDivisionError but allows
        # denominator to remain arbitrarily close to zero if a_hat_ui << epsilon.
        denominator = a_hat[u] + epsilon

    # Compute serendipity score: relevance penalized by natural discoverability
    S = pref / denominator

    # Re-apply the observed mask to S (pref is already masked, but S may carry
    # residual values from the division; this guarantees clean -inf for all
    # previously seen items)
    S[observed_mask] = -np.inf

    return S



# -----------------------------------------------------------------------------
# SLATE ASSEMBLY VARIANTS
# -----------------------------------------------------------------------------

def _collect_slate_rows(u: int, scores: np.ndarray, score_col: str,
                        pref_scores: np.ndarray, a_hat: np.ndarray,
                        item_to_category: dict,
                        artists: pd.DataFrame, item2idx: dict,
                        top_indices: np.ndarray) -> list:
    """
    Internal helper: converts an array of top item indices into a list of
    row dicts ready for DataFrame construction in the slate builders.

    Args:
        u (int): Zero-indexed target user.
        scores (np.ndarray): 1D array of shape (n_items,) containing the primary 
            ranking scores (e.g., S_ui or pref_score). Items set to -np.inf are 
            skipped.
        score_col (str): Column name for the primary score in the output dict 
            (e.g., 'S_ui' or 'pref_score').
        pref_scores (np.ndarray): 1D array of shape (n_items,) containing 
            unconfounded preference scores for all items. Stored separately from 
            scores to allow S_ui and pref_score to coexist in the slate.
        a_hat (np.ndarray): Dense exposure probability matrix of shape 
            (n_users, n_items).
        item_to_category (dict): Maps item index (int) --> category string.
        artists (pd.DataFrame): Artist metadata for readable name lookup.
        item2idx (dict): Maps original artistID --> zero-indexed integer.
        top_indices (np.ndarray): Sorted array of item indices to include, 
            typically the output of np.argsort[::-1].

    Returns:
        list: List of dicts, one per valid item, with keys [artist, category,
            score_col, pref_score, exposure_prob].
    """
    idx2item = {v: k for k, v in item2idx.items()}
    rows = []
    for item_idx in top_indices:
        if scores[item_idx] == -np.inf:
            break
        artist_id = idx2item[item_idx]
        name_series = artists.loc[artists['artistID'] == artist_id, 'name']
        artist_name = (name_series.values[0]
                       if len(name_series) > 0 else f'Artist_{artist_id}')
        rows.append({
            'artist': artist_name,
            'category': item_to_category.get(item_idx, 'other'),
            score_col: scores[item_idx],
            'pref_score': pref_scores[item_idx],
            'exposure_prob': a_hat[u, item_idx],
        })
    return rows


def _to_ranked_df(rows: list, sort_col: str) -> pd.DataFrame:
    """
    Internal helper: converts a list of row dicts into a ranked DataFrame.

    Sorts rows in descending order by sort_col and resets the index to
    start at rank 1.

    Args:
        rows (list): List of row dicts, typically from _collect_slate_rows.
        sort_col (str): Column name to sort by in descending order.

    Returns:
        pd.DataFrame: Ranked slate DataFrame indexed from 1, or an empty 
            DataFrame if rows is empty.
    """
    if not rows:
        return pd.DataFrame()
    df = (pd.DataFrame(rows)
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True))
    df.index += 1
    df.index.name = 'rank'
    return df


def build_std_slate(u: int, model: StandardLMF, a_hat: np.ndarray, 
                    A: csr_matrix, item_to_category: dict, 
                    artists: pd.DataFrame, item2idx: dict, 
                    n_slots: int = N_SLOTS) -> pd.DataFrame:
    """
    Standard LMF slate WITHOUT utility constraints.

    Ranks unobserved items purely by confounded preference score
    (theta_u @ beta_i) and returns the top n_slots globally.

    Args:
        u (int): Zero-indexed target user.
        model (StandardLMF): Trained standard LMF model.
        a_hat (np.ndarray): Exposure probability matrix (n_users, n_items).
        A (csr_matrix): Binary exposure matrix for masking observed items.
        item_to_category (dict): Maps item index --> category string.
        artists (pd.DataFrame): Artist metadata for readable name lookup.
        item2idx (dict): Maps original artistID --> zero-indexed index.
        n_slots (int): Number of recommendation slots. Defaults to N_SLOTS.

    Returns:
        pd.DataFrame: Slate with columns [artist, category, pref_score,
            exposure_prob], indexed from rank 1.
    """
    scores = model.predict(u).copy()
    observed_mask = np.array(A[u].todense()).flatten() > 0
    scores[observed_mask] = -np.inf

    # For Standard LMF, the ranking score and preference score are identical
    pref_scores = scores.copy()
    top_idx = np.argsort(scores)[::-1][:n_slots]
    rows = _collect_slate_rows(u, scores, 'pref_score', pref_scores, a_hat, 
                               item_to_category, artists, item2idx, top_idx)
    return _to_ranked_df(rows, 'pref_score')


def build_causal_slate(u: int, model: CausalLMF, a_hat: np.ndarray, 
                       A: csr_matrix, item_to_category: dict, 
                       artists: pd.DataFrame, item2idx: dict, 
                       n_slots: int = N_SLOTS) -> pd.DataFrame:
    """
    Causal LMF slate WITHOUT utility constraints.

    Ranks unobserved items by unconfounded preference score
    (theta_u @ beta_i, with gamma_u dropped at inference) and returns
    the top n_slots globally.

    Args:
        u (int): Zero-indexed target user.
        model (CausalLMF): Trained causal LMF model.
        a_hat (np.ndarray): Exposure probability matrix (n_users, n_items).
        A (csr_matrix): Binary exposure matrix for masking observed items.
        item_to_category (dict): Maps item index --> category string.
        artists (pd.DataFrame): Artist metadata.
        item2idx (dict): Maps original artistID --> zero-indexed index.
        n_slots (int): Number of recommendation slots. Defaults to N_SLOTS.

    Returns:
        pd.DataFrame: Slate with columns [artist, category, pref_score,
            exposure_prob], indexed from rank 1.
    """
    scores = model.predict_unconfounded(u).copy()
    observed_mask = np.array(A[u].todense()).flatten() > 0
    scores[observed_mask] = -np.inf

    top_idx = np.argsort(scores)[::-1][:n_slots]
    rows = _collect_slate_rows(u, scores, 'pref_score', scores.copy(), a_hat, 
                               item_to_category, artists, item2idx, top_idx)
    return _to_ranked_df(rows, 'pref_score')


def build_serendipity_slate(u: int, model: CausalLMF, a_hat: np.ndarray, 
                            A: csr_matrix, item_to_category: dict,
                            artists: pd.DataFrame, item2idx: dict, 
                            tau_r: float = TAU_R, epsilon: float = EPSILON,
                            winsorize: bool = False,
                            n_slots: int = N_SLOTS) -> pd.DataFrame:
    """
    Causal LMF + Serendipity slate WITHOUT utility constraints.

    Ranks unobserved items by the serendipity score and returns the top
    n_slots globally with no category quotas applied. Supports both the
    original additive-shift denominator and winsorized lower-bound capping.

    Original: S_ui = (theta_u @ beta_i) / (a_hat_ui + epsilon)
    Winsorized: S_ui = (theta_u @ beta_i) / max(a_hat_ui, epsilon)

    Args:
        u (int): Zero-indexed target user.
        model (CausalLMF): Trained causal LMF model.
        a_hat (np.ndarray): Exposure probability matrix (n_users, n_items).
        A (csr_matrix): Binary exposure matrix for masking observed items.
        item_to_category (dict): Maps item index --> category string.
        artists (pd.DataFrame): Artist metadata.
        item2idx (dict): Maps original artistID --> zero-indexed index.
        tau_r (float): Minimum preference score floor. Defaults to TAU_R.
        epsilon (float): Denominator floor. Use a tiny value (e.g. EPSILON) for 
            the original mode, or a meaningful threshold (e.g. WINSORIZE_EPSILON 
            or compute_dynamic_epsilon()) for winsorized mode. Defaults to 
            EPSILON.
        winsorize (bool): If True, apply winsorization instead of additive
            shift. Defaults to False.
        n_slots (int): Number of recommendation slots. Defaults to N_SLOTS.

    Returns:
        pd.DataFrame: Slate with columns [artist, category, S_ui, pref_score, 
            exposure_prob], indexed from rank 1.
    """
    S = compute_serendipity_scores(u, model, a_hat, A, tau_r, epsilon,
                                   winsorize=winsorize)
    pref = model.predict_unconfounded(u)

    top_idx = np.argsort(S)[::-1][:n_slots]
    rows = _collect_slate_rows(u, S, 'S_ui', pref, a_hat, item_to_category, 
                               artists, item2idx, top_idx)
    return _to_ranked_df(rows, 'S_ui')
