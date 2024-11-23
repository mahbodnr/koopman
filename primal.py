import torch
from typing import Tuple
from scipy.linalg import eig

from torch.linalg import eigh
import numpy as np

def weighted_norm(vectors: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the weighted norm of vectors with respect to a weight matrix."""
    return torch.sqrt(torch.sum(vectors * (weight_matrix @ vectors), dim=0))

def eigh_generalized(a: torch.Tensor, b: torch.Tensor, 
                          reg_param: float = 1e-6,
                          normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the generalized eigenvalue problem Ax = λBx using PyTorch with improved numerical stability.
    
    Args:
        a (torch.Tensor): Real symmetric matrix A
        b (torch.Tensor): Real symmetric positive definite matrix B
        reg_param (float): Regularization parameter for numerical stability
        normalize (bool): Whether to normalize eigenvectors
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Eigenvalues and eigenvectors
    """
    # Ensure matrices are symmetric (handle numerical errors)
    a = 0.5 * (a + a.T)
    b = 0.5 * (b + b.T)
    
    # Add small regularization to B to improve conditioning
    b = b + reg_param * torch.eye(b.shape[0], device=b.device, dtype=b.dtype)
    
    try:
        # Try Cholesky first
        l = torch.linalg.cholesky(b)
    except RuntimeError:
        # If Cholesky fails, try adding more regularization
        b = b + reg_param * 10 * torch.eye(b.shape[0], device=b.device, dtype=b.dtype)
        l = torch.linalg.cholesky(b)
    
    # Compute inverse of L carefully
    try:
        l_inv = torch.linalg.inv(l)
    except RuntimeError:
        # If direct inverse fails, use solve with identity matrix
        l_inv = torch.linalg.solve_triangular(l, 
                                            torch.eye(l.shape[0], device=l.device, dtype=l.dtype),
                                            upper=False)
    
    # Transform to standard eigenvalue problem
    c = l_inv @ a @ l_inv.T
    
    # Ensure symmetry of transformed matrix (handle numerical errors)
    c = 0.5 * (c + c.T)
    
    try:
        # Try regular eigh first
        eigvals, eigvecs = torch.linalg.eigh(c)
    except RuntimeError:
        # If it fails, try with increased precision
        c_64 = c.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(c_64)
        eigvals = eigvals.to(c.dtype)
        eigvecs = eigvecs.to(c.dtype)
    
    # Transform eigenvectors back
    eigvecs = l_inv.T @ eigvecs
    
    if normalize:
        # Normalize eigenvectors with respect to B
        # Handle potential numerical issues in normalization
        b_eigvecs = b @ eigvecs
        norms = torch.sqrt(torch.sum(eigvecs * b_eigvecs, dim=0))
        norms = torch.clamp(norms, min=1e-12)  # Prevent division by zero
        eigvecs = eigvecs / norms
    
    return eigvals, eigvecs



def fit_reduced_rank_regression(
    C_X: torch.Tensor,  # Input covariance matrix
    C_XY: torch.Tensor,  # Cross-covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0.0
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. Arnoldi is faster for low ranks.
) -> torch.Tensor:
    """
    Fit reduced rank regression using PyTorch.
    
    Args:
        C_X: Input covariance matrix (torch.Tensor)
        C_XY: Cross-covariance matrix (torch.Tensor)
        tikhonov_reg: Ridge regularization parameter
        rank: Rank of the estimator
        svd_solver: SVD solver method ('arnoldi' or other)
        
    Returns:
        torch.Tensor: The fitted vectors
    """
    if tikhonov_reg == 0.0:
        # Assuming _fit_reduced_rank_regression_noreg is defined elsewhere
        return _fit_reduced_rank_regression_noreg(C_X, C_XY, rank, svd_solver)
    else:
        dim = C_X.size(0)
        reg_input_covariance = C_X + tikhonov_reg * torch.eye(
            dim, dtype=C_X.dtype, device=C_X.device
        )
        _crcov = C_XY @ C_XY.t()
        
        if svd_solver == "arnoldi":
            # For Arnoldi method, we need to use torch.lobpcg or fall back to CPU for scipy
            # Here we'll use torch.lobpcg as it's more native to PyTorch
            guess = torch.randn(dim, rank + 3, device=C_X.device, dtype=C_X.dtype)
            values, vectors = torch.lobpcg(
                _crcov, 
                guess, 
                B=reg_input_covariance,
                largest=True,
                method="ortho"
            )
        else:
            # Using torch.linalg.eigh for the general case
            values, vectors = eigh_generalized(
                _crcov, 
                reg_input_covariance
            )
            # Sort in descending order since eigh returns ascending
            values = values.flip(0)
            vectors = vectors.flip(1)

        # Select top k eigenvalues/vectors
        values = values[:rank]
        vectors = vectors[:, :rank]

        # Normalize vectors
        _norms = weighted_norm(vectors, reg_input_covariance)
        vectors = vectors @ torch.diag(_norms.pow(-1.0))
        
        return vectors


import torch

def predict(
    num_steps: int,  # Number of steps to predict (return the last one)
    U: torch.Tensor,  # Projection matrix, as returned by the fit functions
    C_XY: torch.Tensor,  # Cross-covariance matrix
    phi_Xin: torch.Tensor,  # Feature map evaluated on the initial conditions
    phi_X: torch.Tensor,  # Feature map evaluated on the training input data
    obs_train_Y: torch.Tensor,  # Observable to be predicted evaluated on the output training data
) -> torch.Tensor:
    """
    Predict future states using reduced rank regression in PyTorch.
    
    Args:
        num_steps: Number of steps to predict
        U: Projection matrix from fit
        C_XY: Cross-covariance matrix
        phi_Xin: Initial condition features
        phi_X: Training input features
        obs_train_Y: Training output observations
        
    Returns:
        torch.Tensor: Predicted values
    """
    num_train = phi_X.size(0)
    
    # Compute intermediate matrices
    phi_Xin_dot_U = phi_Xin @ U
    U_C_XY_U = U.t() @ C_XY @ U
    U_phi_X_obs_Y = U.t() @ phi_X.t() @ obs_train_Y * (num_train**-1)
    
    # Compute matrix power
    M = torch.matrix_power(U_C_XY_U, num_steps - 1)
    
    # Final multiplication
    return phi_Xin_dot_U @ M @ U_phi_X_obs_Y



def estimator_eig(
    U: np.ndarray,  # Projection matrix, as returned by the fit functions defined above
    C_XY: np.ndarray,  # Cross-covariance matrix
):
    # Using the trick described in https://arxiv.org/abs/1905.11490
    U = U.numpy()
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = eig(M, left=True, right=True)

    values = fuzzy_parse_complex(values)

    r_perm = np.argsort(values)
    l_perm = np.argsort(values.conj())
    values = values[r_perm]

    # Normalization in RKHS norm
    rv = U @ rv
    rv = rv[:, r_perm]
    rv = rv / np.linalg.norm(rv, axis=0)
    # Biorthogonalization
    lv = np.linalg.multi_dot([C_XY.T, U, lv])
    lv = lv[:, l_perm]
    l_norm = np.sum(lv * rv, axis=0)
    lv = lv / l_norm

    return values, lv, rv