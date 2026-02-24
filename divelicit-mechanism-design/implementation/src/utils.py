import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def psd_check(M: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix M is positive semi-definite."""
    if M.shape[0] != M.shape[1]:
        return False
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals >= -tol))


def nearest_psd(M: np.ndarray) -> np.ndarray:
    """Project matrix M to nearest PSD matrix via eigenvalue clipping."""
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def log_det_safe(M: np.ndarray, eps: float = 1e-10) -> float:
    """Numerically stable log-determinant."""
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, eps)
    return float(np.sum(np.log(eigvals)))
