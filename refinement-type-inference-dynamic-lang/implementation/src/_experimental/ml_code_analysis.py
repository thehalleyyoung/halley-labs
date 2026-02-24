"""ML / data-science code bug detection.

Detects data leakage, train/test contamination, tensor shape errors,
gradient bugs (PyTorch), device mismatches, reproducibility issues,
and scikit-learn API misuse — all via AST analysis.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class DataLeakageKind(Enum):
    FIT_BEFORE_SPLIT = "fit_before_split"
    SCALER_ON_FULL = "scaler_on_full"
    FEATURE_FROM_TARGET = "feature_from_target"
    FUTURE_DATA_LEAK = "future_data_leak"
    TARGET_LEAK = "target_leak"


@dataclass
class DataLeakage:
    kind: DataLeakageKind
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


@dataclass
class Contamination:
    message: str
    line: int
    column: int
    train_var: str
    test_var: str
    severity: str = "error"
    confidence: float = 0.85

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [contamination] {self.message}"


class ShapeErrorKind(Enum):
    BROADCAST_MISMATCH = "broadcast_mismatch"
    MATMUL_MISMATCH = "matmul_mismatch"
    CONCAT_MISMATCH = "concat_mismatch"
    RESHAPE_INVALID = "reshape_invalid"
    INDEX_OUT_OF_DIMS = "index_out_of_dims"
    SQUEEZE_NO_DIM = "squeeze_no_dim"


@dataclass
class ShapeError:
    kind: ShapeErrorKind
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.7
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


class GradientBugKind(Enum):
    MISSING_ZERO_GRAD = "missing_zero_grad"
    MISSING_DETACH = "missing_detach"
    MISSING_NO_GRAD = "missing_no_grad"
    REQUIRES_GRAD_WRONG = "requires_grad_wrong"
    DETACH_BEFORE_BACKWARD = "detach_before_backward"
    DOUBLE_BACKWARD = "double_backward"
    GRAD_IN_EVAL = "grad_in_eval"
    IN_PLACE_ON_LEAF = "in_place_on_leaf"


@dataclass
class GradientBug:
    kind: GradientBugKind
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


@dataclass
class DeviceMismatch:
    message: str
    line: int
    column: int
    devices: List[str] = field(default_factory=list)
    severity: str = "error"
    confidence: float = 0.75
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [device_mismatch] {self.message}"


class ReproIssueKind(Enum):
    MISSING_SEED = "missing_seed"
    NON_DETERMINISTIC_OP = "non_deterministic_op"
    MISSING_WORKER_SEED = "missing_worker_seed"
    CUDNN_NON_DETERMINISTIC = "cudnn_non_deterministic"
    SHUFFLE_WITHOUT_SEED = "shuffle_without_seed"


@dataclass
class ReproIssue:
    kind: ReproIssueKind
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.7
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


class SklearnBugKind(Enum):
    FIT_TRANSFORM_ON_TEST = "fit_transform_on_test"
    PREDICT_BEFORE_FIT = "predict_before_fit"
    WRONG_SCORING = "wrong_scoring"
    MISSING_PIPELINE = "missing_pipeline"
    FEATURE_NAMES_MISMATCH = "feature_names_mismatch"
    CROSS_VAL_LEAKAGE = "cross_val_leakage"
    OVERFIT_NO_REGULARIZATION = "overfit_no_regularization"


@dataclass
class SklearnBug:
    kind: SklearnBugKind
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.75
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _name_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


SCALER_CLASSES: Set[str] = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer",
    "LabelEncoder", "OrdinalEncoder", "OneHotEncoder",
    "TfidfVectorizer", "CountVectorizer", "HashingVectorizer",
}

SKLEARN_MODELS: Set[str] = {
    "LinearRegression", "LogisticRegression", "Ridge", "Lasso",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "SVR", "SVC", "KNeighborsClassifier", "KNeighborsRegressor",
    "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
    "MLPClassifier", "MLPRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "KMeans", "DBSCAN", "AgglomerativeClustering",
}


# ── Data leakage detection ──────────────────────────────────────────────────

def detect_data_leakage(source: str) -> List[DataLeakage]:
    """Detect data leakage patterns in ML code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[DataLeakage] = []
    scaler_vars: Dict[str, int] = {}
    split_line: Optional[int] = None
    fit_lines: Dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            base = name.split(".")[-1] if "." in name else name
            if base in SCALER_CLASSES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        scaler_vars[target.id] = node.lineno

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if "train_test_split" in name:
                split_line = node.lineno

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("fit", "fit_transform"):
                obj_name = _name_str(node.func.value)
                if obj_name in scaler_vars:
                    fit_lines[obj_name] = node.lineno
                    if split_line and node.lineno < split_line:
                        bugs.append(DataLeakage(
                            kind=DataLeakageKind.FIT_BEFORE_SPLIT,
                            message=(
                                f"Scaler '{obj_name}' fit before train/test split at line {split_line}; "
                                f"this leaks test set statistics into the model"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Move fit() after train_test_split() and use fit_transform() only on train data",
                        ))
                    if node.func.attr == "fit_transform" and node.args:
                        arg_name = _name_str(node.args[0]) if node.args else ""
                        if arg_name and not any(
                            kw in arg_name.lower()
                            for kw in ("train", "x_train", "trn")
                        ):
                            bugs.append(DataLeakage(
                                kind=DataLeakageKind.SCALER_ON_FULL,
                                message=f"fit_transform() called on '{arg_name}' — may include test data",
                                line=node.lineno,
                                column=node.col_offset,
                                confidence=0.6,
                                fix_suggestion="Use fit_transform() on training data only, transform() on test data",
                            ))

    return bugs


# ── Train/test contamination ────────────────────────────────────────────────

def detect_train_test_contamination(source: str) -> List[Contamination]:
    """Detect train/test contamination patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[Contamination] = []
    train_vars: Set[str] = set()
    test_vars: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    low = target.id.lower()
                    if "train" in low or "trn" in low:
                        train_vars.add(target.id)
                    elif "test" in low or "val" in low or "valid" in low:
                        test_vars.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            low = elt.id.lower()
                            if "train" in low or "trn" in low:
                                train_vars.add(elt.id)
                            elif "test" in low or "val" in low or "valid" in low:
                                test_vars.add(elt.id)

    # Check for fit_transform on test vars
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "fit_transform" and node.args:
                arg = _name_str(node.args[0])
                if arg in test_vars:
                    bugs.append(Contamination(
                        message=f"fit_transform() called on test variable '{arg}' — use transform() instead",
                        line=node.lineno,
                        column=node.col_offset,
                        train_var="",
                        test_var=arg,
                        fix_suggestion="Use scaler.transform(test_data) instead of fit_transform()",
                    ))

    # Check for concatenation of train and test before fit
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("pd.concat", "np.concatenate", "np.vstack", "np.hstack",
                        "pd.DataFrame.append", "torch.cat"):
                has_train = False
                has_test = False
                for arg in node.args:
                    for child in ast.walk(arg):
                        if isinstance(child, ast.Name):
                            if child.id in train_vars:
                                has_train = True
                            if child.id in test_vars:
                                has_test = True
                if has_train and has_test:
                    bugs.append(Contamination(
                        message=f"Concatenating train and test data via '{name}' before processing",
                        line=node.lineno,
                        column=node.col_offset,
                        train_var="<train>",
                        test_var="<test>",
                        fix_suggestion="Process train and test data separately",
                    ))

    return bugs


# ── Tensor shape error detection ─────────────────────────────────────────────

def detect_tensor_shape_errors(source: str) -> List[ShapeError]:
    """Detect potential tensor/array shape errors."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[ShapeError] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)

            # torch.matmul / np.dot / @ operator dimension checks
            if name in ("torch.matmul", "torch.mm", "np.dot", "np.matmul"):
                if len(node.args) >= 2:
                    bugs.append(ShapeError(
                        kind=ShapeErrorKind.MATMUL_MISMATCH,
                        message=f"Verify dimensions are compatible for {name}()",
                        line=node.lineno,
                        column=node.col_offset,
                        confidence=0.4,
                        fix_suggestion="Ensure inner dimensions match: (m,k) @ (k,n)",
                    ))

            # reshape with -1 validation
            if name.endswith(".reshape") or name.endswith(".view"):
                if node.args:
                    has_multiple_neg = sum(
                        1 for a in node.args
                        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub)
                        and isinstance(a.operand, ast.Constant) and a.operand.value == 1
                    )
                    if has_multiple_neg > 1:
                        bugs.append(ShapeError(
                            kind=ShapeErrorKind.RESHAPE_INVALID,
                            message=f"Multiple -1 dimensions in {name}() — only one allowed",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.95,
                        ))

            # torch.cat / np.concatenate without axis
            if name in ("torch.cat", "torch.stack", "np.concatenate", "np.stack"):
                has_dim_kwarg = any(
                    kw.arg in ("dim", "axis") for kw in node.keywords
                )
                if not has_dim_kwarg and len(node.args) <= 1:
                    bugs.append(ShapeError(
                        kind=ShapeErrorKind.CONCAT_MISMATCH,
                        message=f"{name}() without explicit dim/axis — default may not match intent",
                        line=node.lineno,
                        column=node.col_offset,
                        confidence=0.5,
                        severity="warning",
                        fix_suggestion=f"Add explicit dim= or axis= argument to {name}()",
                    ))

            # squeeze without dim
            if name.endswith(".squeeze"):
                if not node.args and not node.keywords:
                    bugs.append(ShapeError(
                        kind=ShapeErrorKind.SQUEEZE_NO_DIM,
                        message=f".squeeze() without dim — may remove unexpected dimensions",
                        line=node.lineno,
                        column=node.col_offset,
                        confidence=0.6,
                        severity="warning",
                        fix_suggestion="Specify dim= to only squeeze the intended dimension",
                    ))

        # @ operator (matmul)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            bugs.append(ShapeError(
                kind=ShapeErrorKind.MATMUL_MISMATCH,
                message="Matrix multiplication (@) — verify inner dimensions match",
                line=node.lineno,
                column=node.col_offset,
                confidence=0.3,
            ))

    return bugs


# ── Gradient bug detection ──────────────────────────────────────────────────

def detect_gradient_bugs(source: str) -> List[GradientBug]:
    """Detect PyTorch gradient bugs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[GradientBug] = []
    has_backward = False
    has_zero_grad = False
    has_no_grad_context = False
    optimizer_vars: Set[str] = set()
    model_vars: Set[str] = set()
    in_eval_mode: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            base = name.split(".")[-1] if "." in name else name
            if base in ("Adam", "SGD", "AdamW", "RMSprop", "Adagrad", "Adadelta"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        optimizer_vars.add(target.id)
            if base in ("Module", "Sequential", "Linear", "Conv2d", "LSTM", "GRU",
                        "Transformer", "BatchNorm2d", "LayerNorm"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        model_vars.add(target.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_has_backward = False
            func_has_zero_grad = False
            func_has_step = False

            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    method = child.func.attr
                    obj = _name_str(child.func.value)

                    if method == "backward":
                        func_has_backward = True
                        has_backward = True

                    if method == "zero_grad":
                        func_has_zero_grad = True
                        has_zero_grad = True

                    if method == "step" and obj in optimizer_vars:
                        func_has_step = True

                    if method == "eval":
                        in_eval_mode.add(obj)

                    if method == "train":
                        in_eval_mode.discard(obj)

                    # In-place operations on tensors
                    if method.endswith("_") and method not in ("zero_", "fill_", "normal_", "uniform_"):
                        bugs.append(GradientBug(
                            kind=GradientBugKind.IN_PLACE_ON_LEAF,
                            message=f"In-place operation '.{method}()' may cause gradient computation issues",
                            line=child.lineno,
                            column=child.col_offset,
                            confidence=0.5,
                            severity="warning",
                            fix_suggestion="Use out-of-place version to avoid gradient errors",
                        ))

            if func_has_backward and not func_has_zero_grad:
                bugs.append(GradientBug(
                    kind=GradientBugKind.MISSING_ZERO_GRAD,
                    message=f"backward() called in '{node.name}' without zero_grad() — gradients will accumulate",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="error",
                    confidence=0.85,
                    fix_suggestion="Add optimizer.zero_grad() before loss.backward()",
                ))

    # Check for missing torch.no_grad() in eval/inference
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name_lower = node.name.lower()
            is_inference = any(kw in name_lower for kw in ("eval", "predict", "infer", "test", "validate"))
            if is_inference:
                has_no_grad = False
                has_forward = False
                for child in ast.walk(node):
                    if isinstance(child, ast.With):
                        for item in child.items:
                            ctx = _name_str(item.context_expr)
                            if "no_grad" in ctx:
                                has_no_grad = True
                    if isinstance(child, ast.Call):
                        cn = _get_call_name(child)
                        if cn.endswith("forward") or cn.endswith("__call__"):
                            has_forward = True
                        if isinstance(child.func, ast.Attribute):
                            obj = _name_str(child.func.value)
                            if obj in model_vars:
                                has_forward = True
                if has_forward and not has_no_grad:
                    bugs.append(GradientBug(
                        kind=GradientBugKind.MISSING_NO_GRAD,
                        message=f"Inference function '{node.name}' runs model without torch.no_grad()",
                        line=node.lineno,
                        column=node.col_offset,
                        confidence=0.7,
                        fix_suggestion="Wrap inference code in 'with torch.no_grad():'",
                    ))

    return bugs


# ── Device mismatch detection ───────────────────────────────────────────────

def detect_device_mismatch(source: str) -> List[DeviceMismatch]:
    """Detect tensor device mismatches (CPU vs CUDA)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[DeviceMismatch] = []
    var_devices: Dict[str, Tuple[str, int]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    device = _infer_device(node.value)
                    if device:
                        var_devices[target.id] = (device, node.lineno)

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            left_dev = _expr_device(node.left, var_devices)
            right_dev = _expr_device(node.right, var_devices)
            if left_dev and right_dev and left_dev != right_dev:
                bugs.append(DeviceMismatch(
                    message=f"Operation between tensors on different devices: {left_dev} vs {right_dev}",
                    line=node.lineno,
                    column=node.col_offset,
                    devices=[left_dev, right_dev],
                    fix_suggestion=f"Move tensors to same device: tensor.to('{left_dev}')",
                ))

        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("torch.matmul", "torch.mm", "torch.bmm",
                        "torch.cat", "torch.stack", "torch.add"):
                devices: Set[str] = set()
                for arg in node.args:
                    dev = _expr_device(arg, var_devices)
                    if dev:
                        devices.add(dev)
                if len(devices) > 1:
                    bugs.append(DeviceMismatch(
                        message=f"{name}() with tensors on different devices: {', '.join(devices)}",
                        line=node.lineno,
                        column=node.col_offset,
                        devices=list(devices),
                        fix_suggestion="Move all tensors to the same device before operation",
                    ))

    return bugs


def _infer_device(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Call):
        name = _get_call_name(node)
        if name in ("torch.tensor", "torch.zeros", "torch.ones", "torch.randn",
                     "torch.rand", "torch.empty", "torch.eye"):
            for kw in node.keywords:
                if kw.arg == "device" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
            return "cpu"
        if name.endswith(".to"):
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "device" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
        if name.endswith(".cuda"):
            return "cuda"
        if name.endswith(".cpu"):
            return "cpu"
    return None


def _expr_device(node: ast.expr, var_devices: Dict[str, Tuple[str, int]]) -> Optional[str]:
    if isinstance(node, ast.Name) and node.id in var_devices:
        return var_devices[node.id][0]
    if isinstance(node, ast.Call):
        return _infer_device(node)
    return None


# ── Reproducibility issue detection ─────────────────────────────────────────

def detect_reproducibility_issues(source: str) -> List[ReproIssue]:
    """Detect reproducibility issues in ML code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[ReproIssue] = []
    has_seed_set = False
    has_torch_manual_seed = False
    has_np_random_seed = False
    has_cudnn_deterministic = False
    uses_torch = False
    uses_numpy = False
    uses_random = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)

            if name == "torch.manual_seed":
                has_torch_manual_seed = True
            if name in ("np.random.seed", "numpy.random.seed"):
                has_np_random_seed = True
            if name == "random.seed":
                has_seed_set = True
            if name.startswith("torch."):
                uses_torch = True
            if name.startswith("np.") or name.startswith("numpy."):
                uses_numpy = True
            if name.startswith("random."):
                uses_random = True

            # torch.backends.cudnn.deterministic
            if name == "torch.use_deterministic_algorithms":
                has_cudnn_deterministic = True

        if isinstance(node, ast.Assign):
            for target in node.targets:
                tgt = _name_str(target)
                if "cudnn.deterministic" in tgt:
                    has_cudnn_deterministic = True

    if uses_torch and not has_torch_manual_seed:
        bugs.append(ReproIssue(
            kind=ReproIssueKind.MISSING_SEED,
            message="PyTorch used without torch.manual_seed() — results may not be reproducible",
            line=1,
            column=0,
            fix_suggestion="Add torch.manual_seed(42) at the beginning of training",
        ))

    if uses_numpy and not has_np_random_seed:
        bugs.append(ReproIssue(
            kind=ReproIssueKind.MISSING_SEED,
            message="NumPy used without np.random.seed() — results may not be reproducible",
            line=1,
            column=0,
            fix_suggestion="Add np.random.seed(42) at the beginning",
        ))

    if uses_random and not has_seed_set:
        bugs.append(ReproIssue(
            kind=ReproIssueKind.MISSING_SEED,
            message="random module used without random.seed() — results may not be reproducible",
            line=1,
            column=0,
            fix_suggestion="Add random.seed(42) at the beginning",
        ))

    if uses_torch and not has_cudnn_deterministic:
        bugs.append(ReproIssue(
            kind=ReproIssueKind.CUDNN_NON_DETERMINISTIC,
            message="torch.backends.cudnn.deterministic not set — CuDNN may use non-deterministic algorithms",
            line=1,
            column=0,
            confidence=0.5,
            fix_suggestion="Set torch.backends.cudnn.deterministic = True",
        ))

    # Check for DataLoader without worker seed
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("DataLoader", "torch.utils.data.DataLoader"):
                has_num_workers = False
                has_worker_init = False
                for kw in node.keywords:
                    if kw.arg == "num_workers":
                        if isinstance(kw.value, ast.Constant) and kw.value.value and kw.value.value > 0:
                            has_num_workers = True
                    if kw.arg == "worker_init_fn":
                        has_worker_init = True
                if has_num_workers and not has_worker_init:
                    bugs.append(ReproIssue(
                        kind=ReproIssueKind.MISSING_WORKER_SEED,
                        message="DataLoader with num_workers>0 but no worker_init_fn — workers may produce different data",
                        line=node.lineno,
                        column=node.col_offset,
                        fix_suggestion="Add worker_init_fn to seed each worker deterministically",
                    ))

            if name in ("random.shuffle", "np.random.shuffle"):
                bugs.append(ReproIssue(
                    kind=ReproIssueKind.SHUFFLE_WITHOUT_SEED,
                    message=f"{name}() called — ensure random seed is set for reproducibility",
                    line=node.lineno,
                    column=node.col_offset,
                    confidence=0.5,
                ))

    return bugs


# ── scikit-learn API misuse ─────────────────────────────────────────────────

def sklearn_api_misuse(source: str) -> List[SklearnBug]:
    """Detect scikit-learn API misuse patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[SklearnBug] = []
    model_vars: Dict[str, int] = {}
    fitted_models: Set[str] = set()
    scaler_vars: Dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            base = name.split(".")[-1] if "." in name else name
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if base in SKLEARN_MODELS:
                        model_vars[target.id] = node.lineno
                    if base in SCALER_CLASSES:
                        scaler_vars[target.id] = node.lineno

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            obj = _name_str(node.func.value)
            method = node.func.attr

            if method == "fit" and obj in model_vars:
                fitted_models.add(obj)
            if method == "fit_transform" and obj in scaler_vars:
                fitted_models.add(obj)

            if method == "predict" and obj in model_vars:
                if obj not in fitted_models:
                    bugs.append(SklearnBug(
                        kind=SklearnBugKind.PREDICT_BEFORE_FIT,
                        message=f"predict() called on '{obj}' which may not have been fitted",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.7,
                        fix_suggestion=f"Call {obj}.fit(X_train, y_train) before predict()",
                    ))

            if method == "fit_transform" and obj in scaler_vars:
                if node.args:
                    arg_name = _name_str(node.args[0])
                    low = arg_name.lower()
                    if any(kw in low for kw in ("test", "val", "valid")):
                        bugs.append(SklearnBug(
                            kind=SklearnBugKind.FIT_TRANSFORM_ON_TEST,
                            message=f"fit_transform() on test data '{arg_name}' — use transform() instead",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.9,
                            fix_suggestion=f"Use {obj}.transform({arg_name}) for test/validation data",
                        ))

    # Check for lack of Pipeline usage when scaler + model exist
    if scaler_vars and model_vars:
        has_pipeline = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_call_name(node)
                if "Pipeline" in name or "make_pipeline" in name:
                    has_pipeline = True
                    break
        if not has_pipeline:
            bugs.append(SklearnBug(
                kind=SklearnBugKind.MISSING_PIPELINE,
                message="Scaler and model used without Pipeline — risk of data leakage in cross-validation",
                line=1,
                column=0,
                confidence=0.6,
                fix_suggestion="Use sklearn.pipeline.Pipeline([('scaler', scaler), ('model', model)])",
            ))

    return bugs
