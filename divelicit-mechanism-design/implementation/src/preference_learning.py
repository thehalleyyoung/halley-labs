"""
Preference Learning Module — discrete choice models, active elicitation,
rank aggregation, and choice axiom testing. Uses numpy/scipy only.
"""
import numpy as np
from scipy.special import expit, softmax
from scipy.optimize import minimize
from itertools import combinations
from typing import List, Tuple, Optional, Dict, Any


class BradleyTerryModel:
    """Bradley-Terry model for pairwise comparisons via MLE (Newton's method).
    P(i > j) = sigma(lambda_i - lambda_j) where sigma is the logistic function."""

    def __init__(self, n_items: int):
        """Initialize with n_items alternatives."""
        self.n_items = n_items
        self.params = np.zeros(n_items)
        self.comparisons: List[Tuple[int, int]] = []

    def add_comparison(self, winner: int, loser: int) -> None:
        """Record that winner was preferred over loser."""
        self.comparisons.append((winner, loser))

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Compute log-likelihood of observed comparisons."""
        ll = 0.0
        for w, l in self.comparisons:
            diff = params[w] - params[l]
            ll += diff - np.log1p(np.exp(diff))
        return ll

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """Gradient of the log-likelihood."""
        grad = np.zeros(self.n_items)
        for w, l in self.comparisons:
            p = expit(params[w] - params[l])
            grad[w] += 1.0 - p
            grad[l] -= 1.0 - p
        return grad

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """Hessian of the log-likelihood."""
        H = np.zeros((self.n_items, self.n_items))
        for w, l in self.comparisons:
            p = expit(params[w] - params[l])
            wt = p * (1.0 - p)
            H[w, w] -= wt
            H[l, l] -= wt
            H[w, l] += wt
            H[l, w] += wt
        return H

    def fit(self, max_iter: int = 100, tol: float = 1e-8, reg: float = 1e-4) -> np.ndarray:
        """Fit via Newton's method with Tikhonov regularization. Pin params[0]=0."""
        params = np.zeros(self.n_items)
        for _ in range(max_iter):
            grad = self._gradient(params) - reg * params
            H = self._hessian(params) - reg * np.eye(self.n_items)
            grad[0] = 0.0
            H[0, :] = 0.0; H[:, 0] = 0.0; H[0, 0] = -1.0
            try:
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                delta = -0.1 * grad
            step = 1.0
            cur_ll = self._log_likelihood(params)
            for _ in range(20):
                if self._log_likelihood(params + step * delta) > cur_ll - 1e-4 * step * grad @ delta:
                    break
                step *= 0.5
            params += step * delta
            params[0] = 0.0
            if np.linalg.norm(grad[1:]) < tol:
                break
        self.params = params
        return params

    def predict_prob(self, i: int, j: int) -> float:
        """P(i > j) under fitted model."""
        return float(expit(self.params[i] - self.params[j]))

    def rank_items(self) -> np.ndarray:
        """Items sorted by decreasing strength."""
        return np.argsort(-self.params)


class PlackettLuceModel:
    """Plackett-Luce model for ranking data, fitted via Hunter's MM algorithm.
    P(ranking) = prod_{j} theta_{sigma_j} / sum_{l>=j} theta_{sigma_l}."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.params = np.ones(n_items) / n_items
        self.rankings: List[List[int]] = []

    def add_ranking(self, ranking: List[int]) -> None:
        """Add a full or partial ranking (best to worst)."""
        self.rankings.append(list(ranking))

    def _log_likelihood(self, params: np.ndarray) -> float:
        ll = 0.0
        for ranking in self.rankings:
            for j in range(len(ranking) - 1):
                denom = np.sum(params[ranking[j:]])
                if denom > 0:
                    ll += np.log(params[ranking[j]]) - np.log(denom)
        return ll

    def fit(self, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
        """Fit via the MM algorithm of Hunter (2004)."""
        params = np.ones(self.n_items, dtype=float)
        params /= params.sum()
        for _ in range(max_iter):
            num = np.zeros(self.n_items)
            den = np.zeros(self.n_items)
            for ranking in self.rankings:
                k = len(ranking)
                suffix = np.zeros(k)
                suffix[k - 1] = params[ranking[k - 1]]
                for j in range(k - 2, -1, -1):
                    suffix[j] = suffix[j + 1] + params[ranking[j]]
                for j in range(k - 1):
                    num[ranking[j]] += 1.0
                for j in range(k):
                    for pos in range(j + 1):
                        if suffix[pos] > 0:
                            den[ranking[j]] += 1.0 / suffix[pos]
            new_p = np.where(den > 0, num / den, params)
            s = new_p.sum()
            if s > 0:
                new_p /= s
            if np.max(np.abs(new_p - params)) < tol:
                break
            params = new_p
        self.params = params
        return params

    def predict_ranking(self, items: Optional[List[int]] = None) -> List[int]:
        """Predict most likely ranking of items (or all items if None)."""
        if items is None:
            items = list(range(self.n_items))
        order = np.argsort(-self.params[items])
        return [items[i] for i in order]

    def sample_ranking(self, items: Optional[List[int]] = None,
                       rng: Optional[np.random.Generator] = None) -> List[int]:
        """Sample a ranking from the PL distribution."""
        if rng is None:
            rng = np.random.default_rng()
        if items is None:
            items = list(range(self.n_items))
        remaining = list(items)
        ranking = []
        for _ in range(len(items)):
            probs = self.params[remaining]
            total = probs.sum()
            if total <= 0:
                ranking.extend(remaining)
                break
            idx = rng.choice(len(remaining), p=probs / total)
            ranking.append(remaining[idx])
            remaining.pop(idx)
        return ranking


class RandomUtilityModel:
    """Random Utility Models: logit (Gumbel noise) and probit (Gaussian noise).
    U_i = v_i + epsilon_i; choice = argmax U_i over the choice set."""

    def __init__(self, n_items: int, model_type: str = "logit"):
        if model_type not in ("logit", "probit"):
            raise ValueError("model_type must be 'logit' or 'probit'")
        self.n_items = n_items
        self.model_type = model_type
        self.utilities = np.zeros(n_items)
        self.covariance = np.eye(n_items)
        self.observations: List[Tuple[List[int], int]] = []

    def add_observation(self, choice_set: List[int], chosen: int) -> None:
        """Record that chosen was selected from choice_set."""
        self.observations.append((list(choice_set), chosen))

    def _logit_log_likelihood(self, v: np.ndarray) -> float:
        """Log-likelihood under multinomial logit."""
        ll = 0.0
        for cs, chosen in self.observations:
            u = v[cs]
            mx = np.max(u)
            ll += v[chosen] - mx - np.log(np.sum(np.exp(u - mx)))
        return ll

    def _probit_choice_prob(self, v: np.ndarray, cs: List[int],
                            chosen: int, n_sim: int = 5000,
                            rng: Optional[np.random.Generator] = None) -> float:
        """Estimate choice probability under probit via Monte Carlo."""
        if rng is None:
            rng = np.random.default_rng(42)
        sub_mean = v[cs]
        sub_cov = self.covariance[np.ix_(cs, cs)]
        try:
            draws = rng.multivariate_normal(sub_mean, sub_cov, size=n_sim)
        except np.linalg.LinAlgError:
            draws = rng.multivariate_normal(sub_mean, sub_cov + 1e-6 * np.eye(len(cs)), size=n_sim)
        chosen_idx = cs.index(chosen)
        return max(float(np.mean(np.argmax(draws, axis=1) == chosen_idx)), 1e-10)

    def fit(self, max_iter: int = 200, tol: float = 1e-6, reg: float = 1e-3) -> np.ndarray:
        """Fit utilities via MLE. Logit uses L-BFGS-B; probit uses Nelder-Mead."""
        if self.model_type == "logit":
            def neg_ll(v):
                return -self._logit_log_likelihood(v) + 0.5 * reg * v @ v
            def neg_grad(v):
                g = np.zeros(self.n_items)
                for cs, chosen in self.observations:
                    probs = softmax(v[cs])
                    for k, item in enumerate(cs):
                        g[item] -= probs[k]
                    g[chosen] += 1.0
                return -g + reg * v
            res = minimize(neg_ll, self.utilities, jac=neg_grad, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol})
            self.utilities = res.x
        else:
            rng = np.random.default_rng(42)
            def neg_sim_ll(v):
                ll = sum(np.log(self._probit_choice_prob(v, cs, ch, 2000, rng))
                         for cs, ch in self.observations)
                return -ll + 0.5 * reg * v @ v
            res = minimize(neg_sim_ll, self.utilities, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': tol})
            self.utilities = res.x
        return self.utilities

    def predict_choice(self, choice_set: List[int]) -> int:
        """Predict the most likely choice from a set."""
        return choice_set[int(np.argmax(self.utilities[choice_set]))]

    def choice_probabilities(self, choice_set: List[int]) -> np.ndarray:
        """Compute choice probabilities for all items in the set."""
        if self.model_type == "logit":
            return softmax(self.utilities[choice_set])
        probs = np.array([self._probit_choice_prob(self.utilities, choice_set, it, 5000)
                          for it in choice_set])
        s = probs.sum()
        return probs / s if s > 0 else probs


class ActivePreferenceLearner:
    """Bayesian active learning for pairwise preference queries.
    Selects pairs maximizing expected information gain under a
    Gaussian-approximated posterior (Laplace approximation)."""

    def __init__(self, n_items: int, prior_var: float = 1.0):
        self.n_items = n_items
        self.mean = np.zeros(n_items)
        self.covariance = prior_var * np.eye(n_items)
        self.comparisons: List[Tuple[int, int]] = []

    def _posterior_update(self, winner: int, loser: int) -> None:
        """Update posterior via Laplace approximation after observing winner > loser."""
        self.comparisons.append((winner, loser))
        prior_prec = np.linalg.inv(self.covariance)
        def neg_lp(p):
            d = p - self.mean
            val = 0.5 * d @ prior_prec @ d
            for w, l in self.comparisons:
                diff = p[w] - p[l]
                val -= diff - np.log1p(np.exp(diff))
            return val
        def grad_neg_lp(p):
            g = prior_prec @ (p - self.mean)
            for w, l in self.comparisons:
                r = 1.0 - expit(p[w] - p[l])
                g[w] -= r; g[l] += r
            return g
        res = minimize(neg_lp, self.mean, jac=grad_neg_lp, method='L-BFGS-B',
                       options={'maxiter': 100})
        self.mean = res.x
        # Hessian at MAP for covariance
        H = prior_prec.copy()
        for w, l in self.comparisons:
            p = expit(self.mean[w] - self.mean[l])
            wt = p * (1.0 - p)
            H[w, w] += wt; H[l, l] += wt; H[w, l] -= wt; H[l, w] -= wt
        try:
            self.covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.covariance = np.linalg.inv(H + 1e-6 * np.eye(self.n_items))

    def _expected_info_gain(self, i: int, j: int) -> float:
        """Expected reduction in posterior entropy from querying (i, j)."""
        p_i = expit(self.mean[i] - self.mean[j])
        cur_ent = 0.5 * np.linalg.slogdet(self.covariance)[1]
        prior_prec = np.linalg.inv(self.covariance)
        total = 0.0
        for prob in [p_i, 1.0 - p_i]:
            H_new = prior_prec.copy()
            for w, l in self.comparisons:
                p = expit(self.mean[w] - self.mean[l])
                wt = p * (1.0 - p)
                H_new[w, w] += wt; H_new[l, l] += wt
                H_new[w, l] -= wt; H_new[l, w] -= wt
            # Add contribution from the hypothetical new observation
            wt = p_i * (1.0 - p_i)
            H_new[i, i] += wt; H_new[j, j] += wt
            H_new[i, j] -= wt; H_new[j, i] -= wt
            sign, logdet = np.linalg.slogdet(H_new)
            new_ent = -0.5 * logdet if sign > 0 else cur_ent
            total += prob * new_ent
        return cur_ent - total

    def select_query(self) -> Tuple[int, int]:
        """Select the pair with maximum expected information gain."""
        best, best_g = (0, 1), -np.inf
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                g = self._expected_info_gain(i, j)
                if g > best_g:
                    best_g = g; best = (i, j)
        return best

    def observe(self, winner: int, loser: int) -> np.ndarray:
        """Record comparison and return updated mean parameters."""
        self._posterior_update(winner, loser)
        return self.mean.copy()

    def current_ranking(self) -> np.ndarray:
        """Items sorted by decreasing posterior mean strength."""
        return np.argsort(-self.mean)


class RegretMinimizingElicitor:
    """Preference elicitation minimizing minimax regret over a particle-based
    version space of plausible utility vectors."""

    def __init__(self, n_items: int, n_particles: int = 200,
                 rng: Optional[np.random.Generator] = None):
        self.n_items = n_items
        self.n_particles = n_particles
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.particles = self.rng.standard_normal((n_particles, n_items))
        self.weights = np.ones(n_particles) / n_particles

    def _minimax_regret(self, rec: int) -> float:
        """Max weighted regret of recommending item rec."""
        best = np.max(self.particles, axis=1)
        regrets = best - self.particles[:, rec]
        return float(np.max(regrets * self.weights * self.n_particles))

    def recommend(self) -> int:
        """Item with minimum maximum regret."""
        return int(min(range(self.n_items), key=self._minimax_regret))

    def select_query(self) -> Tuple[int, int]:
        """Select pair minimizing expected post-query minimax regret."""
        best_pair, best_er = (0, 1), np.inf
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                mask = self.particles[:, i] > self.particles[:, j]
                p_i = np.sum(self.weights[mask])
                old_w = self.weights.copy()
                # Simulate i-wins
                wi = old_w.copy(); wi[~mask] *= 0.1
                s = wi.sum()
                if s > 0: wi /= s
                self.weights = wi
                ri = min(self._minimax_regret(k) for k in range(self.n_items))
                # Simulate j-wins
                wj = old_w.copy(); wj[mask] *= 0.1
                s = wj.sum()
                if s > 0: wj /= s
                self.weights = wj
                rj = min(self._minimax_regret(k) for k in range(self.n_items))
                self.weights = old_w
                er = p_i * ri + (1 - p_i) * rj
                if er < best_er:
                    best_er = er; best_pair = (i, j)
        return best_pair

    def observe(self, winner: int, loser: int) -> None:
        """Update version space weights after observing winner > loser."""
        consistent = self.particles[:, winner] > self.particles[:, loser]
        self.weights[consistent] *= 2.0
        self.weights[~consistent] *= 0.1
        s = self.weights.sum()
        if s > 0:
            self.weights /= s
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 3:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling with jitter for particle diversity."""
        cs = np.cumsum(self.weights); cs[-1] = 1.0
        u = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        idx = np.clip(np.searchsorted(cs, u), 0, self.n_particles - 1)
        self.particles = self.particles[idx].copy()
        self.particles += 0.05 * self.rng.standard_normal(self.particles.shape)
        self.weights = np.ones(self.n_particles) / self.n_particles


class RankAggregator:
    """Rank aggregation: Borda count, Copeland, and Kemeny-optimal (branch & bound)."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.rankings: List[List[int]] = []

    def add_ranking(self, ranking: List[int]) -> None:
        """Add a ranking (best to worst) to aggregate."""
        self.rankings.append(list(ranking))

    def _pairwise_matrix(self) -> np.ndarray:
        """M[i][j] = number of rankings where i is above j."""
        M = np.zeros((self.n_items, self.n_items))
        for ranking in self.rankings:
            for i in range(len(ranking)):
                for j in range(i + 1, len(ranking)):
                    M[ranking[i], ranking[j]] += 1
        return M

    def borda_count(self) -> List[int]:
        """Aggregate via Borda count (points = items ranked below)."""
        scores = np.zeros(self.n_items)
        for ranking in self.rankings:
            k = len(ranking)
            for pos, item in enumerate(ranking):
                scores[item] += (k - 1 - pos)
        return list(np.argsort(-scores))

    def copeland(self) -> List[int]:
        """Aggregate via Copeland's method (+1 pairwise majority win, -1 loss)."""
        M = self._pairwise_matrix()
        scores = np.zeros(self.n_items)
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if M[i, j] > M[j, i]:
                    scores[i] += 1; scores[j] -= 1
                elif M[j, i] > M[i, j]:
                    scores[j] += 1; scores[i] -= 1
        return list(np.argsort(-scores))

    def _kendall_tau(self, r1: List[int], r2: List[int]) -> int:
        """Kendall tau distance (number of discordant pairs)."""
        items = set(r1) & set(r2)
        pos1 = {it: i for i, it in enumerate(r1) if it in items}
        pos2 = {it: i for i, it in enumerate(r2) if it in items}
        its = sorted(items)
        dist = 0
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                a, b = its[i], its[j]
                if (pos1[a] - pos1[b]) * (pos2[a] - pos2[b]) < 0:
                    dist += 1
        return dist

    def _total_kendall(self, candidate: List[int]) -> int:
        """Total Kendall distance from candidate to all input rankings."""
        return sum(self._kendall_tau(candidate, r) for r in self.rankings)

    def kemeny_optimal(self, max_items: int = 8) -> List[int]:
        """Find Kemeny optimal ranking via branch and bound.
        Minimizes total Kendall tau distance to all input rankings."""
        items = list(range(min(self.n_items, max_items)))
        if len(items) <= 1:
            return items
        M = self._pairwise_matrix()
        best_ranking = list(self.borda_count()[:len(items)])
        best_dist = self._total_kendall(best_ranking)

        def lower_bound(partial, remaining):
            lb = 0
            for r in self.rankings:
                pos_r = {it: i for i, it in enumerate(r) if it in set(items)}
                for a in partial:
                    for b in remaining:
                        if a in pos_r and b in pos_r and pos_r[a] > pos_r[b]:
                            lb += 1
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    a, b = remaining[i], remaining[j]
                    lb += min(M[a, b], M[b, a])
            return lb

        def bb(partial, remaining, cur_dist):
            nonlocal best_ranking, best_dist
            if not remaining:
                if cur_dist < best_dist:
                    best_dist = cur_dist; best_ranking = list(partial)
                return
            if cur_dist + lower_bound(partial, remaining) >= best_dist:
                return
            order = sorted(remaining, key=lambda x: -sum(M[x, r] for r in remaining if r != x))
            for item in order:
                nd = cur_dist
                all_items_set = set(items)
                for r in self.rankings:
                    pos_r = {it: i for i, it in enumerate(r) if it in all_items_set}
                    if item in pos_r:
                        for prev in partial:
                            if prev in pos_r and pos_r[prev] > pos_r[item]:
                                nd += 1
                bb(partial + [item], [x for x in remaining if x != item], nd)

        bb([], items, 0)
        return best_ranking


class ChoiceAxiomTester:
    """Tests IIA, regularity, and rationalizability (WARP) on choice data."""

    def __init__(self):
        self.observations: Dict[frozenset, Dict[int, int]] = {}

    def add_observation(self, choice_set: List[int], chosen: int) -> None:
        """Record that chosen was selected from choice_set."""
        key = frozenset(choice_set)
        if key not in self.observations:
            self.observations[key] = {}
        self.observations[key][chosen] = self.observations[key].get(chosen, 0) + 1

    def _choice_freq(self, cs: frozenset) -> Dict[int, float]:
        """Empirical choice frequencies for a choice set."""
        counts = self.observations.get(cs, {})
        total = sum(counts.values())
        return {it: c / total for it, c in counts.items()} if total > 0 else {}

    def test_iia(self, significance: float = 0.05) -> Dict[str, Any]:
        """Test Independence of Irrelevant Alternatives (Luce's axiom).
        Checks that P(a|S)/P(b|S) is constant across all sets containing a,b."""
        all_pairs: Dict[Tuple[int, int], List[float]] = {}
        for cs, counts in self.observations.items():
            total = sum(counts.values())
            if total < 2:
                continue
            items = list(counts.keys())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = (items[i], items[j]) if items[i] < items[j] else (items[j], items[i])
                    fb = counts.get(b, 0) / total
                    if fb > 0:
                        ratio = (counts.get(a, 0) / total) / fb
                        all_pairs.setdefault((a, b), []).append(ratio)
        violations = []
        max_var = 0.0
        for (a, b), ratios in all_pairs.items():
            finite = [r for r in ratios if np.isfinite(r)]
            if len(finite) < 2:
                continue
            arr = np.array(finite)
            mu = np.mean(arr)
            cv = np.std(arr) / mu if mu > 0 else 0.0
            if cv > significance:
                violations.append((a, b, cv))
            max_var = max(max_var, cv)
        return {'iia_holds': len(violations) == 0, 'violations': violations,
                'max_ratio_variation': max_var}

    def test_regularity(self) -> Dict[str, Any]:
        """Test regularity: S ⊂ T => P(a|T) <= P(a|S) for all a in S."""
        violations = []
        sets = list(self.observations.keys())
        for i in range(len(sets)):
            for j in range(len(sets)):
                if i == j or not sets[i].issubset(sets[j]):
                    continue
                fS = self._choice_freq(sets[i])
                fT = self._choice_freq(sets[j])
                for item in sets[i]:
                    pS, pT = fS.get(item, 0.0), fT.get(item, 0.0)
                    if pT > pS + 1e-10:
                        violations.append({'item': item, 'subset': set(sets[i]),
                                           'superset': set(sets[j]),
                                           'prob_subset': pS, 'prob_superset': pT})
        return {'regularity_holds': len(violations) == 0, 'violations': violations}

    def test_rationalizability(self) -> Dict[str, Any]:
        """Test WARP: if a chosen over b, then b never chosen when a available.
        Also detects cycles in revealed preference via Floyd-Warshall."""
        revealed: Dict[int, set] = {}
        for cs, counts in self.observations.items():
            if not counts:
                continue
            best = max(counts.keys(), key=lambda x: counts[x])
            revealed.setdefault(best, set())
            for item in cs:
                if item != best:
                    revealed[best].add(item)
        # WARP violations
        all_items = set()
        for it, worse in revealed.items():
            all_items.add(it); all_items.update(worse)
        warp_v = []
        for a in all_items:
            for b in all_items:
                if a < b and b in revealed.get(a, set()) and a in revealed.get(b, set()):
                    warp_v.append((a, b))
        # Cycle detection via Floyd-Warshall
        its = sorted(all_items)
        n = len(its)
        idx = {it: i for i, it in enumerate(its)}
        adj = np.zeros((n, n), dtype=bool)
        for it, worse in revealed.items():
            if it in idx:
                for w in worse:
                    if w in idx:
                        adj[idx[it], idx[w]] = True
        reach = adj.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if reach[i, k] and reach[k, j]:
                        reach[i, j] = True
        cycles = []
        for i in range(n):
            if reach[i, i]:
                members = sorted([its[i]] + [its[j] for j in range(n)
                                             if j != i and reach[i, j] and reach[j, i]])
                if members not in cycles:
                    cycles.append(members)
        return {'rationalizable': len(warp_v) == 0 and len(cycles) == 0,
                'warp_violations': warp_v, 'cycles': cycles}


class PreferenceLearner:
    """Unified preference learning interface wrapping BT, PL, and RUM models.
    Provides observe/predict API, active query selection, and axiom testing."""

    def __init__(self, n_items: int, model: str = "bradley-terry"):
        """Initialize with n_items and model type
        ('bradley-terry', 'plackett-luce', 'logit', 'probit')."""
        self.n_items = n_items
        self.model_name = model
        self.axiom_tester = ChoiceAxiomTester()
        if model == "bradley-terry":
            self._bt = BradleyTerryModel(n_items)
            self._active = ActivePreferenceLearner(n_items)
        elif model == "plackett-luce":
            self._pl = PlackettLuceModel(n_items)
        elif model in ("logit", "probit"):
            self._rum = RandomUtilityModel(n_items, model_type=model)
        else:
            raise ValueError(f"Unknown model: {model}")

    def observe(self, choice_set: List[int], chosen: int) -> np.ndarray:
        """Observe a choice from choice_set and update the model. Returns parameters."""
        self.axiom_tester.add_observation(choice_set, chosen)
        if self.model_name == "bradley-terry":
            for item in choice_set:
                if item != chosen:
                    self._bt.add_comparison(chosen, item)
            params = self._bt.fit()
            other = choice_set[0] if choice_set[0] != chosen else choice_set[1]
            self._active.observe(chosen, other)
            return params
        elif self.model_name == "plackett-luce":
            partial = [chosen] + [x for x in choice_set if x != chosen]
            self._pl.add_ranking(partial)
            return self._pl.fit()
        else:
            self._rum.add_observation(choice_set, chosen)
            return self._rum.fit()

    def predict(self, choice_set: List[int]) -> List[int]:
        """Predict ranking of items in the choice set (best to worst)."""
        if self.model_name == "bradley-terry":
            order = np.argsort(-self._bt.params[choice_set])
            return [choice_set[i] for i in order]
        elif self.model_name == "plackett-luce":
            return self._pl.predict_ranking(choice_set)
        else:
            order = np.argsort(-self._rum.utilities[choice_set])
            return [choice_set[i] for i in order]

    def select_query(self) -> Tuple[int, int]:
        """Select most informative pairwise query via active learning."""
        if self.model_name == "bradley-terry":
            return self._active.select_query()
        if self.model_name == "plackett-luce":
            params = self._pl.params
        else:
            params = np.exp(self._rum.utilities)
        best, best_e = (0, 1), -1.0
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                p = params[i] / (params[i] + params[j] + 1e-30)
                e = -(p * np.log(p + 1e-30) + (1 - p) * np.log(1 - p + 1e-30))
                if e > best_e:
                    best_e = e; best = (i, j)
        return best

    def test_axioms(self) -> Dict[str, Any]:
        """Run IIA, regularity, and rationalizability tests on accumulated data."""
        return {'iia': self.axiom_tester.test_iia(),
                'regularity': self.axiom_tester.test_regularity(),
                'rationalizability': self.axiom_tester.test_rationalizability()}

    def goodness_of_fit(self, test_data: List[Tuple[List[int], int]]) -> Dict[str, float]:
        """Evaluate model on held-out data. Returns log_likelihood, accuracy, top2_accuracy."""
        ll, correct, top2 = 0.0, 0, 0
        for cs, chosen in test_data:
            ranking = self.predict(cs)
            if ranking[0] == chosen:
                correct += 1
            if chosen in ranking[:2]:
                top2 += 1
            if self.model_name == "bradley-terry":
                probs = softmax(self._bt.params[cs])
                p = max(probs[cs.index(chosen)], 1e-15)
            elif self.model_name == "plackett-luce":
                pm = self._pl.params[cs]
                s = pm.sum()
                p = max(pm[cs.index(chosen)] / s, 1e-15) if s > 0 else 1e-15
            else:
                probs = self._rum.choice_probabilities(cs)
                p = max(probs[cs.index(chosen)], 1e-15)
            ll += np.log(p)
        n = max(len(test_data), 1)
        return {'log_likelihood': ll, 'accuracy': correct / n, 'top2_accuracy': top2 / n}
