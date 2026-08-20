"""Falsification tests for the fictitious-play route to Nash on the finite grid.

The proposed route is that Exp3 behaves like a perturbed fictitious play, that
its limit points are Nash equilibria of the finite-grid game, and that those
converge to the continuous competitive equilibrium as the tick shrinks. These
tests look for a counterexample rather than for further confirmation: a dynamic
that sustains market prices a fixed distance from the benchmark.

The quantity every test reports is the market error

    D(q_1, q_2) = |min(ask) - a*| + |max(bid) - b*|,

the distance of the prices a trader can actually reach from the benchmark. It is
deliberately not the distance of an individual quote, since a maker can sit far
away on the side it never trades while the realized prices are correct.

Tests
-----
    A  pure best-response graph: any sink component without a Nash action
    B  mixed Nash equilibria and their market error
    C  exact fictitious play under four tie-breaking rules
    D  deterministic smoothed fictitious play
    E  the existing stochastic AgentExp3MeanBased

Usage
-----
    python scripts/exp3_falsification.py --self-test
    python scripts/exp3_falsification.py --tests A,B --deltas 0.125
    python scripts/exp3_falsification.py --rounds 1000000

Neither learner is modified by this script.
"""

import argparse
import math
import traceback

import matplotlib.pyplot as plt
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import lama_lab.analysis as analysis
from lama_lab.agents import AgentExp3MeanBased
from lama_lab.analysis import get_ask, get_bid
from lama_lab.diagnostics import Exp3Diagnostics, build_log_checkpoints
from lama_lab.envs import MarketMakingEnvironment
from lama_lab.generators import UniformGenerator
from lama_lab.utils import ResultsManager, setup_logger

EPS = 1e-03
BR_TOL = 1e-12
THRESHOLDS = (0.05, 0.10, 0.20)
TAILS = (0.50, 0.25, 0.10)

# Colours kept apart under colour vision deficiency
PALETTE = ("#1f77b4", "#ff7f0e", "#9467bd", "#8a8a8a")

TIE_RULES = (
    "random_best_response",
    "stay_if_best",
    "widest_quote_among_best",
    "max_distance_from_continuous_q_star_among_best",
)

SCHEDULES = {
    "theory_schedule": (1 / 3, 1 / 4),
    "empirical_default_control": (1 / 2, 2 / 5),
}

SUMMARY_FIELDS = (
    "algorithm",
    "schedule",
    "delta",
    "K",
    "seed",
    "initialization",
    "tie_break_rule",
    "T_final",
    "final_current_expected_market_error",
    "final_bad_market_mass_eps_005",
    "final_bad_market_mass_eps_010",
    "final_bad_market_mass_eps_020",
    "cesaro_market_error",
    "fraction_bad_market_eps_005",
    "fraction_bad_market_eps_010",
    "fraction_bad_market_eps_020",
    "tail_mean_market_error",
    "tail_bad_market_mass_eps_010",
    "tail_product_exploitability",
    "product_exploitability",
    "max_external_regret",
    "independence_TV",
    "policy_L1_distance",
    "support_br_gap",
    "worst_active_gap_tau_1e3",
    "worst_active_gap_tau_1e4",
    "argmax_action_player_1",
    "argmax_action_player_2",
    "max_probability_player_1",
    "max_probability_player_2",
    "p1_q_star",
    "p2_q_star",
    "p1_ask_special",
    "p2_ask_special",
    "p1_bid_special",
    "p2_bid_special",
    "p1_withdrawn",
    "p2_withdrawn",
)


# ---------------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------------
def build_game(delta: float, samples: torch.Tensor) -> dict:
    """Assemble everything the tests need for one tick size.

    Parameters
    ----------
    delta : float
        Tick size of the quote grid.
    samples : torch.Tensor
        Draws of the latent value, used for the exact payoff matrix and for the
        continuous benchmark, so that both rest on the same distribution.

    Returns
    -------
    game : dict
        The arms, the payoff matrix, the market-error matrix, the benchmark, the
        pure Nash profiles, the rationalizable layers and the indices of the four
        special actions.
    """
    arms = analysis.build_quote_grid(0.0, 1.0, delta, epsilon=EPS)
    payoff = analysis.get_expected_payoff_matrix(arms, samples, epsilon=EPS)

    fixed_points = analysis.get_all_unique_fixed_points(samples, eps=EPS, tol=EPS)
    b_star, m_star, a_star = fixed_points[0].tolist()

    error = analysis.get_market_error(arms, a_star, b_star)
    nash = analysis.get_pure_nash(payoff)
    layers, _ = analysis.get_rationalizable_set(payoff)

    symmetric = sorted({i for i, j in nash.tolist() if i == j})
    dagger = max(symmetric, key=lambda a: payoff[a, a].item()) if symmetric else None

    return {
        "delta": delta,
        "arms": arms,
        "n_arms": arms.shape[0],
        "payoff": payoff,
        "error": error,
        "benchmark": {"a_star": a_star, "b_star": b_star, "m_star": m_star},
        "nash": nash,
        "nash_actions": symmetric,
        "payoff_dominant_action": dagger,
        "layers": layers,
        "special": find_special_actions(arms, a_star, b_star, delta),
    }


def find_action(
    arms: torch.Tensor,
    bid: float,
    ask: float,
    tol: float,
) -> int | None:
    """Index of the grid quote nearest a target, or ``None`` when none is close.

    The benchmark comes from the empirical distribution, so it misses the grid by
    the sampling error even when the exact value lies on it. The nearest quote
    within `tol` is therefore the right target, and `tol` should be about half a
    tick so that only a genuinely absent quote returns ``None``.
    """
    distance = (get_bid(arms) - bid).abs() + (get_ask(arms) - ask).abs()
    return int(distance.argmin()) if distance.min().item() <= tol else None


def find_special_actions(
    arms: torch.Tensor,
    a_star: float,
    b_star: float,
    delta: float,
) -> dict:
    """Locate the four quotes the passive-coordinate conjecture is about.

    The quotes are built from explicit bid and ask values rather than from tuple
    literals, since the theory writes a quote ask first and the code stores it
    bid first.
    """
    tol = 0.51 * delta
    return {
        "q_star": find_action(arms, b_star, a_star, tol),
        "ask_special": find_action(arms, 0.0, a_star, tol),
        "bid_special": find_action(arms, b_star, 1.0, tol),
        "withdrawn": find_action(arms, 0.0, 1.0, tol),
    }


def market_stats(
    policy_1: torch.Tensor, policy_2: torch.Tensor, error: torch.Tensor
) -> dict:
    """Expected market error and bad-market mass of a product of policies.

    Parameters
    ----------
    policy_1, policy_2 : torch.Tensor
        Policies of shape ``(..., n_arms)``.
    error : torch.Tensor
        Market-error matrix of shape ``(n_arms, n_arms)``.

    Returns
    -------
    stats : dict
        ``expected`` market error and one ``bad_<eps>`` entry per threshold, each
        of shape ``(...)``.
    """
    stats = {"expected": torch.einsum("...i,ij,...j->...", policy_1, error, policy_2)}

    for threshold in THRESHOLDS:
        indicator = (error > threshold).to(error.dtype)
        stats[f"bad_{threshold:g}"] = torch.einsum(
            "...i,ij,...j->...", policy_1, indicator, policy_2
        )
    return stats


def exploitability_of(
    payoff: torch.Tensor, policy_1: torch.Tensor, policy_2: torch.Tensor
):
    """Largest best-response gain of either maker against a product of policies."""
    first, second = analysis.get_exploitability(payoff, policy_1, policy_2)
    return torch.maximum(first, second)


def best_response_diagnostics(
    payoff: torch.Tensor,
    average_opponent: torch.Tensor,
    policy: torch.Tensor,
    taus: tuple[float, ...] = (1e-03, 1e-04),
) -> dict:
    """How close a policy is to best responding to the opponent's average play.

    Tests the approximate-best-response idea directly, rather than inferring it
    from vanishing regret.
    """
    value = average_opponent @ payoff.T
    best = value.amax(dim=-1)

    gaps = {
        "best_response_action": value.argmax(dim=-1),
        "best_response_value": best,
        "support_br_gap": best - (policy * value).sum(dim=-1),
    }

    for tau in taus:
        loss = torch.where(
            policy >= tau, best.unsqueeze(-1) - value, torch.zeros_like(value)
        )
        gaps[f"worst_active_gap_tau_{tau:g}"] = loss.amax(dim=-1)
    return gaps


# ---------------------------------------------------------------------------
# Test A: pure best-response graph
# ---------------------------------------------------------------------------
def test_best_response_graph(game: dict, logger) -> dict:
    """Test A: look for a closed best-response component holding no Nash action.

    The graph has an edge from an opponent action to every exact best response
    against it. A sink strongly connected component is closed under best
    response, so play that enters it never leaves; one containing no Nash action
    would be a counterexample to the route.
    """
    payoff, error, arms = game["payoff"], game["error"], game["arms"]
    n_arms = game["n_arms"]

    best = payoff.amax(dim=0)
    # edges[r, q] is True when q is a best response to r
    edges = (payoff >= best.unsqueeze(0) - BR_TOL).T

    count, labels = connected_components(
        csr_matrix(edges.cpu().numpy()), directed=True, connection="strong"
    )
    labels = torch.as_tensor(labels)

    nash_actions = set(game["nash_actions"])
    components = []

    for component in range(count):
        members = torch.nonzero(labels == component).flatten()
        # A sink has no edge leaving the component
        leaves = bool(edges[members][:, labels != component].any())
        if leaves:
            continue

        inside = error[members][:, members]
        members_list = members.tolist()
        components.append(
            {
                "actions": members_list,
                "quotes": [
                    [get_bid(arms[a]).item(), get_ask(arms[a]).item()]
                    for a in members_list
                ],
                "contains_nash_action": bool(nash_actions & set(members_list)),
                "all_actions_are_nash_actions": set(members_list) <= nash_actions,
                "min_ask": get_ask(arms[members]).min().item(),
                "max_ask": get_ask(arms[members]).max().item(),
                "min_bid": get_bid(arms[members]).min().item(),
                "max_bid": get_bid(arms[members]).max().item(),
                "max_pairwise_market_error": inside.max().item(),
                "size": len(members_list),
            }
        )

    self_loops = [a for a in range(n_arms) if edges[a, a]]
    warnings = [c for c in components if not c["contains_nash_action"]]

    logger.info(
        f"  Test A: {count} components, {len(components)} sinks, "
        f"{len(self_loops)} self loops, {len(warnings)} sinks without a Nash action"
    )

    return {
        "n_components": int(count),
        "sink_components": components,
        "self_loops": self_loops,
        "sinks_without_nash": warnings,
        "special_action_membership": {
            name: (None if index is None else int(labels[index]))
            for name, index in game["special"].items()
        },
    }


# ---------------------------------------------------------------------------
# Test B: mixed Nash equilibria
# ---------------------------------------------------------------------------
def test_mixed_nash(game: dict, max_support: int | None, logger) -> dict:
    """Test B: mixed Nash equilibria and the market error each sustains."""
    payoff, error = game["payoff"], game["error"]
    equilibria = analysis.get_mixed_nash(payoff, max_support=max_support)

    records = []
    for equilibrium in equilibria:
        strategy = torch.tensor(equilibrium["strategy"], dtype=payoff.dtype)
        stats = market_stats(strategy, strategy, error)

        records.append(
            {
                "support_player_1": equilibrium["support"],
                "support_player_2": equilibrium["support"],
                "probabilities": [
                    equilibrium["strategy"][k] for k in equilibrium["support"]
                ],
                "exploitability": exploitability_of(payoff, strategy, strategy).item(),
                "expected_market_error": stats["expected"].item(),
                **{
                    f"bad_market_mass_eps_{int(t * 1000):03d}": stats[
                        f"bad_{t:g}"
                    ].item()
                    for t in THRESHOLDS
                },
            }
        )

    errors = [r["expected_market_error"] for r in records]
    logger.info(
        f"  Test B: {len(records)} symmetric equilibria, market error in "
        f"[{min(errors):.4f}, {max(errors):.4f}]"
        if records
        else "  Test B: none found"
    )

    return {
        "searched_max_support": (
            max_support if max_support is not None else game["n_arms"]
        ),
        "exhaustive": max_support is None or max_support >= game["n_arms"],
        "symmetric_only": True,
        "equilibria": records,
        "min_expected_market_error": min(errors) if errors else float("nan"),
        "max_expected_market_error": max(errors) if errors else float("nan"),
    }


# ---------------------------------------------------------------------------
# Test C: exact fictitious play
# ---------------------------------------------------------------------------
def _tie_break(scores, rule, previous, priority, generator):
    """Choose among the exact best responses according to one rule."""
    best = scores.amax(dim=-1, keepdim=True)
    mask = scores >= best - BR_TOL

    if rule == "random_best_response":
        return torch.multinomial(mask.to(scores.dtype), 1, generator=generator).squeeze(
            -1
        )

    if rule == "stay_if_best":
        first = mask.to(scores.dtype).argmax(dim=-1)
        if previous is None:
            return first
        keeps = mask.gather(-1, previous.unsqueeze(-1)).squeeze(-1)
        return torch.where(keeps, previous, first)

    ranked = torch.where(
        mask, priority.expand_as(scores), torch.full_like(scores, -math.inf)
    )
    return ranked.argmax(dim=-1)


def run_fictitious_play(
    game: dict,
    priors: torch.Tensor,
    rule: str,
    n_rounds: int,
    checkpoints: list[int],
    seed: int,
    logger,
) -> dict:
    """Test C: exact discrete fictitious play, batched over initial conditions.

    Both makers keep cumulative counterfactual scores against the exact payoff
    matrix and play a best response to them. Every initial condition advances in
    the same loop, which is what makes a million rounds affordable.

    Parameters
    ----------
    game : dict
        As returned by :func:`build_game`.
    priors : torch.Tensor
        Initial empirical distributions of shape ``(2, n_init, n_arms)``, worth
        one round of observation each.
    rule : str
        Tie-breaking rule, one of :data:`TIE_RULES`.
    n_rounds : int
        Horizon.
    checkpoints : list of int
        Rounds at which to record diagnostics.
    seed : int
        Seed for the random tie-breaking rule.

    Returns
    -------
    history : dict
        Per-checkpoint diagnostics, each of shape ``(n_checkpoints, n_init)``.
    """
    payoff, error = game["payoff"], game["error"]
    arms, n_arms = game["arms"], game["n_arms"]
    n_init = priors.shape[1]

    generator = torch.Generator(device=payoff.device).manual_seed(seed)
    dtype = payoff.dtype

    if rule == "widest_quote_among_best":
        priority = (get_ask(arms) - get_bid(arms)).to(dtype)
    elif rule == "max_distance_from_continuous_q_star_among_best":
        priority = (
            (get_ask(arms) - game["benchmark"]["a_star"]).abs()
            + (get_bid(arms) - game["benchmark"]["b_star"]).abs()
        ).to(dtype)
    else:
        priority = torch.zeros(n_arms, dtype=dtype)

    scores = torch.einsum("pbj,qj->pbq", priors.to(dtype), payoff)
    counts = torch.zeros((2, n_init, n_arms), dtype=dtype)
    policy_sum = torch.zeros((2, n_init, n_arms), dtype=dtype)

    running = {"error": torch.zeros(n_init, dtype=dtype)}
    for threshold in THRESHOLDS:
        running[f"bad_{threshold:g}"] = torch.zeros(n_init, dtype=dtype)

    tails = {t: torch.zeros(n_init, dtype=dtype) for t in TAILS}
    tail_starts = {t: int(n_rounds * (1.0 - t)) for t in TAILS}

    previous = None
    history = {"round": []}

    for step in range(n_rounds):
        current = torch.stack(
            [
                _tie_break(
                    scores[p],
                    rule,
                    None if previous is None else previous[p],
                    priority,
                    generator,
                )
                for p in range(2)
            ]
        )
        previous = current

        counts[0].scatter_add_(
            1, current[0].unsqueeze(-1), torch.ones((n_init, 1), dtype=dtype)
        )
        counts[1].scatter_add_(
            1, current[1].unsqueeze(-1), torch.ones((n_init, 1), dtype=dtype)
        )
        policy_sum = counts

        realized = error[current[0], current[1]]
        running["error"] += realized
        for threshold in THRESHOLDS:
            running[f"bad_{threshold:g}"] += (realized > threshold).to(dtype)
        for fraction, start in tail_starts.items():
            if step >= start:
                tails[fraction] += realized

        scores[0] += payoff[:, current[1]].T
        scores[1] += payoff[:, current[0]].T

        if (step + 1) in checkpoints:
            done = step + 1
            empirical = policy_sum / done
            record = {
                "round": done,
                "current_action_player_1": current[0].clone(),
                "current_action_player_2": current[1].clone(),
                "current_market_error": realized.clone(),
                "cesaro_market_error": running["error"] / done,
                "product_exploitability": exploitability_of(
                    payoff, empirical[0], empirical[1]
                ),
            }
            for threshold in THRESHOLDS:
                record[f"fraction_bad_{threshold:g}"] = (
                    running[f"bad_{threshold:g}"] / done
                )

            for player in range(2):
                gaps = best_response_diagnostics(
                    payoff, empirical[1 - player], empirical[player]
                )
                record[f"support_br_gap_{player + 1}"] = gaps["support_br_gap"]

            for name, value in record.items():
                history.setdefault(name, []).append(value)

    history["empirical_policy"] = policy_sum / n_rounds
    history["tail_mean_market_error"] = {
        f"{int(f * 100)}": (tails[f] / max(1, n_rounds - tail_starts[f])) for f in TAILS
    }
    logger.info(f"  Test C [{rule}]: {n_init} initialisations, {n_rounds} rounds")
    return history


def build_fictitious_play_priors(
    game: dict, n_random: int, seed: int
) -> tuple[torch.Tensor, list[str]]:
    """Initial empirical distributions for fictitious play."""
    n_arms = game["n_arms"]
    dtype = game["payoff"].dtype
    generator = torch.Generator().manual_seed(seed)

    priors, labels = [], []

    def add(first: torch.Tensor, second: torch.Tensor, label: str) -> None:
        priors.append(torch.stack([first, second]))
        labels.append(label)

    uniform = torch.full((n_arms,), 1.0 / n_arms, dtype=dtype)
    add(uniform, uniform, "uniform")

    for name, index in game["special"].items():
        if index is None:
            continue
        point = torch.zeros(n_arms, dtype=dtype)
        point[index] = 1.0
        add(point, point, name)

    ask_special, bid_special = (
        game["special"]["ask_special"],
        game["special"]["bid_special"],
    )
    if ask_special is not None and bid_special is not None:
        first = torch.zeros(n_arms, dtype=dtype)
        second = torch.zeros(n_arms, dtype=dtype)
        first[ask_special] = 1.0
        second[bid_special] = 1.0
        add(first, second, "ask_special_vs_bid_special")

    for action in game["nash_actions"]:
        point = torch.full((n_arms,), 0.1 / n_arms, dtype=dtype)
        point[action] += 0.9
        point = point / point.sum()
        add(point, point, f"near_nash_{action}")

    concentration = torch.ones(n_arms, dtype=dtype)
    for draw in range(n_random):
        sample = torch._standard_gamma(concentration, generator=generator)
        sample = sample / sample.sum()
        other = torch._standard_gamma(concentration, generator=generator)
        other = other / other.sum()
        add(sample, other, f"dirichlet_1_{draw}")

    return torch.stack(priors, dim=1), labels


# ---------------------------------------------------------------------------
# Test D: deterministic smoothed fictitious play
# ---------------------------------------------------------------------------
def run_smoothed_play(
    game: dict,
    scores: torch.Tensor,
    schedule: tuple[float, float],
    n_rounds: int,
    checkpoints: list[int],
    logger,
) -> dict:
    """Test D: exponential weights on exact expected payoffs, no sampling noise.

    Closest deterministic relative of the stochastic learner: same schedule and
    same explicit exploration, but the score of every action is updated against
    the opponent's whole mixed strategy instead of a sampled reward.
    """
    payoff, error = game["payoff"], game["error"]
    n_arms = game["n_arms"]
    alpha, beta = schedule
    n_init = scores.shape[1]

    policy_sum = torch.zeros_like(scores)
    tails = {t: torch.zeros(n_init, dtype=payoff.dtype) for t in TAILS}
    tail_starts = {t: int(n_rounds * (1.0 - t)) for t in TAILS}
    history = {}

    for step in range(n_rounds):
        t = step + 1
        eta, epsilon = float(t) ** -alpha, float(t) ** -beta

        exploitation = torch.softmax(eta * scores, dim=-1)
        policy = (1.0 - epsilon) * exploitation + epsilon / n_arms
        policy_sum += policy

        stats = market_stats(policy[0], policy[1], error)
        for fraction, start in tail_starts.items():
            if step >= start:
                tails[fraction] += stats["expected"]

        if t in checkpoints:
            average = policy_sum / t
            record = {
                "round": t,
                "eta": eta,
                "epsilon": epsilon,
                "expected_market_error": stats["expected"],
                "product_exploitability": exploitability_of(
                    payoff, policy[0], policy[1]
                ),
                "policy_l1_distance": (policy[0] - policy[1]).abs().sum(-1),
                "entropy_1": torch.special.entr(policy[0]).sum(-1),
                "entropy_2": torch.special.entr(policy[1]).sum(-1),
                "max_probability_1": policy[0].amax(-1),
                "max_probability_2": policy[1].amax(-1),
                "argmax_1": policy[0].argmax(-1),
                "argmax_2": policy[1].argmax(-1),
            }
            for threshold in THRESHOLDS:
                record[f"bad_{threshold:g}"] = stats[f"bad_{threshold:g}"]

            for player in range(2):
                gaps = best_response_diagnostics(
                    payoff, average[1 - player], exploitation[player]
                )
                record[f"support_br_gap_{player + 1}"] = gaps["support_br_gap"]
                record[f"worst_active_gap_tau_1e-3_{player + 1}"] = gaps[
                    "worst_active_gap_tau_0.001"
                ]
                record[f"worst_active_gap_tau_1e-4_{player + 1}"] = gaps[
                    "worst_active_gap_tau_0.0001"
                ]

            record.update(special_action_masses(game, policy))
            for name, value in record.items():
                history.setdefault(name, []).append(value)

        scores[0] += policy[1] @ payoff.T
        scores[1] += policy[0] @ payoff.T

    history["final_policy"] = torch.softmax(float(n_rounds) ** -alpha * scores, dim=-1)
    history["tail_mean_market_error"] = {
        f"{int(f * 100)}": (tails[f] / max(1, n_rounds - tail_starts[f])) for f in TAILS
    }
    logger.info(f"  Test D: {n_init} initialisations, {n_rounds} rounds")
    return history


def special_action_masses(game: dict, policy: torch.Tensor) -> dict:
    """Individual and paired probabilities on the four special quotes.

    The conjecture is that off-market quotes may keep individual probability
    while the pairings of them that produce bad prices vanish, so the pair
    masses matter more than the individual ones.
    """
    special, error = game["special"], game["error"]
    masses = {}

    for name, index in special.items():
        if index is None:
            masses[f"p1_{name}"] = torch.full_like(policy[0, ..., 0], float("nan"))
            masses[f"p2_{name}"] = torch.full_like(policy[0, ..., 0], float("nan"))
            continue
        masses[f"p1_{name}"] = policy[0, ..., index]
        masses[f"p2_{name}"] = policy[1, ..., index]

    pairs = (
        ("ask_special", "ask_special"),
        ("bid_special", "bid_special"),
        ("ask_special", "bid_special"),
        ("bid_special", "ask_special"),
    )
    for first, second in pairs:
        i, j = special[first], special[second]
        key = f"pair_{first}_{second}"
        if i is None or j is None:
            masses[key] = torch.full_like(policy[0, ..., 0], float("nan"))
            masses[f"{key}_error"] = float("nan")
            continue
        masses[key] = policy[0, ..., i] * policy[1, ..., j]
        masses[f"{key}_error"] = error[i, j].item()
    return masses


def build_smoothed_scores(
    game: dict, n_random: int, seed: int
) -> tuple[torch.Tensor, list[str]]:
    """Initial score vectors for smoothed play."""
    n_arms = game["n_arms"]
    dtype = game["payoff"].dtype
    torch.manual_seed(seed)

    scores, labels = [], []

    def add(first, second, label):
        scores.append(torch.stack([first, second]))
        labels.append(label)

    zero = torch.zeros(n_arms, dtype=dtype)
    add(zero.clone(), zero.clone(), "uniform")
    add(0.01 * torch.rand(n_arms, dtype=dtype), zero.clone(), "asymmetric_perturbation")

    for name, index in game["special"].items():
        if index is None:
            continue
        point = zero.clone()
        point[index] = 1.0
        add(point.clone(), point.clone(), f"near_{name}")

    for action in game["nash_actions"]:
        point = zero.clone()
        point[action] = 1.0
        add(point.clone(), point.clone(), f"near_nash_{action}")

    for draw in range(n_random):
        noise = 0.01 * torch.rand(n_arms, dtype=dtype)
        add(noise, noise.clone(), f"random_perturbation_{draw}")

    return torch.stack(scores, dim=1), labels


# ---------------------------------------------------------------------------
# Test E: the existing stochastic learner
# ---------------------------------------------------------------------------
def run_exp3(
    game: dict,
    schedule: tuple[float, float],
    n_seeds: int,
    n_rounds: int,
    checkpoints: list[int],
    seed: int,
    logger,
) -> dict:
    """Test E: market-error diagnostics on the unmodified stochastic learner.

    Seeds are episodes, since the environment already simulates independent
    replicas, so one run covers every seed. Expected quantities use the product
    of the policies before sampling; realized ones use the drawn actions.
    """
    payoff, error, arms = game["payoff"], game["error"], game["arms"]
    delta = game["delta"]
    alpha, beta = schedule

    torch.manual_seed(seed)
    generator = UniformGenerator(0.0, 1.0)
    env = MarketMakingEnvironment(2, n_seeds, n_rounds, generator, epsilon=EPS)
    makers = [
        AgentExp3MeanBased(
            n_seeds,
            arms,
            reward_range=(-(1.0 - delta), 0.5),
            eta_exponent=alpha,
            exploration_exponent=beta,
        )
        for _ in range(2)
    ]
    diagnostics = Exp3Diagnostics(
        payoff, n_episodes=n_seeds, pure_nash=game["nash"], layers=game["layers"]
    )

    dtype = payoff.dtype
    policy_sum = torch.zeros((2, n_seeds, game["n_arms"]), dtype=dtype)
    running = {"error": torch.zeros(n_seeds, dtype=dtype)}
    for threshold in THRESHOLDS:
        running[f"bad_{threshold:g}"] = torch.zeros(n_seeds, dtype=dtype)

    tails = {t: torch.zeros(n_seeds, dtype=dtype) for t in TAILS}
    tail_starts = {t: int(n_rounds * (1.0 - t)) for t in TAILS}
    history = {}

    for step in range(n_rounds):
        policies = torch.stack([m.get_policy().to(dtype) for m in makers])
        policy_sum += policies

        actions = torch.stack([m.act() for m in makers], dim=1)
        rewards = env.step(actions)
        arm_indices = torch.stack([m.get_last_arms() for m in makers])
        for j, maker in enumerate(makers):
            maker.update(rewards[:, j])

        diagnostics.update(policies, arm_indices)

        realized = error[arm_indices[0], arm_indices[1]]
        running["error"] += realized
        for threshold in THRESHOLDS:
            running[f"bad_{threshold:g}"] += (realized > threshold).to(dtype)
        for fraction, start in tail_starts.items():
            if step >= start:
                tails[fraction] += realized

        if (step + 1) in checkpoints:
            done = step + 1
            snapshot = diagnostics.snapshot()
            stats = market_stats(policies[0], policies[1], error)
            average = policy_sum / done

            record = {
                "round": done,
                "eta": makers[0].get_learning_rate(),
                "epsilon": makers[0].get_exploration(),
                "expected_market_error": stats["expected"],
                "cesaro_market_error": running["error"] / done,
                "product_exploitability": snapshot["max_last_exploitability"],
                "max_avg_regret": snapshot["max_avg_regret"],
                "independence_tv": snapshot["independence_tv"],
                "mass_outside_cr": snapshot["mass_outside_cr_1"],
                "policy_drift": snapshot["policy_drift_1"],
                "policy_l1_distance": (policies[0] - policies[1]).abs().sum(-1),
                "max_probability_1": policies[0].amax(-1),
                "max_probability_2": policies[1].amax(-1),
                "argmax_1": policies[0].argmax(-1),
                "argmax_2": policies[1].argmax(-1),
            }
            for threshold in THRESHOLDS:
                record[f"bad_{threshold:g}"] = stats[f"bad_{threshold:g}"]
                record[f"fraction_bad_{threshold:g}"] = (
                    running[f"bad_{threshold:g}"] / done
                )

            for player in range(2):
                gaps = best_response_diagnostics(
                    payoff, average[1 - player], policies[player]
                )
                record[f"support_br_gap_{player + 1}"] = gaps["support_br_gap"]
                record[f"worst_active_gap_tau_1e-3_{player + 1}"] = gaps[
                    "worst_active_gap_tau_0.001"
                ]
                record[f"worst_active_gap_tau_1e-4_{player + 1}"] = gaps[
                    "worst_active_gap_tau_0.0001"
                ]

            record.update(special_action_masses(game, policies))
            for name, value in record.items():
                history.setdefault(name, []).append(value)

            logger.info(
                f"    Test E round {done}: E[D] median {stats['expected'].median():.5f}"
            )

    history["final_policy"] = torch.stack([m.get_policy().to(dtype) for m in makers])
    history["tail_mean_market_error"] = {
        f"{int(f * 100)}": (tails[f] / max(1, n_rounds - tail_starts[f])) for f in TAILS
    }
    return history


# ---------------------------------------------------------------------------
# Self tests
# ---------------------------------------------------------------------------
def self_test(samples: torch.Tensor, logger) -> None:
    """Checks that must pass before any long run.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    game = build_game(0.125, samples)
    arms, payoff, error = game["arms"], game["payoff"], game["error"]
    benchmark = game["benchmark"]

    # Action ordering: the whole suite is wrong if this is reversed, so the
    # accessors are checked against a quote whose bid and ask are known
    assert bool((get_ask(arms) > get_bid(arms)).all()), "ask must exceed bid"

    probe = torch.tensor([[0.25, 0.75], [0.125, 0.875]], dtype=arms.dtype)
    assert get_bid(probe).tolist() == [0.25, 0.125], "get_bid picked the wrong column"
    assert get_ask(probe).tolist() == [0.75, 0.875], "get_ask picked the wrong column"

    best_ask = get_ask(probe).min().item()
    best_bid = get_bid(probe).max().item()
    midpoint = (best_ask + best_bid) / 2.0
    assert best_ask == 0.75, f"best ask should be min(ask), got {best_ask}"
    assert best_bid == 0.25, f"best bid should be max(bid), got {best_bid}"
    assert abs(midpoint - 0.5) < 1e-12, f"midpoint should be 0.5, got {midpoint}"
    logger.info(
        f"  ordering: best ask {best_ask:.4f}, best bid {best_bid:.4f}, "
        f"midpoint {midpoint:.4f}"
    )

    # Continuous benchmark of the uniform model
    for name, expected in (("a_star", 0.75), ("b_star", 0.25), ("m_star", 0.5)):
        assert abs(benchmark[name] - expected) < 2e-03, f"{name} = {benchmark[name]}"

    # Market-error helper against the values the specification fixes. The
    # benchmark comes from the empirical distribution, so it misses the exact
    # value by the sampling error and these hold to that tolerance, not exactly.
    special = game["special"]
    assert error[special["q_star"], special["q_star"]].item() < 2e-03
    assert (
        abs(
            error[special["ask_special"]][special["ask_special"]].item()
            - benchmark["b_star"]
        )
        < 2e-03
    )
    assert (
        abs(
            error[special["bid_special"]][special["bid_special"]].item()
            - (1.0 - benchmark["a_star"])
        )
        < 2e-03
    )
    assert bool(torch.allclose(error, error.T))

    # One payoff route: the matrix must agree with the environment it models
    env = MarketMakingEnvironment(2, 40_000, 4, UniformGenerator(0.0, 1.0), epsilon=EPS)
    torch.manual_seed(0)
    for first, second in ((0, 5), (3, 3), (7, 1)):
        pairing = torch.stack([arms[first], arms[second]]).to(torch.float32)
        realized = env.step(pairing.unsqueeze(0).expand(40_000, 2, 2))[:, 0]
        error_bar = 5.0 * realized.std().item() / math.sqrt(40_000) + 2e-03
        assert abs(realized.mean().item() - payoff[first, second].item()) < error_bar

    # Fictitious play must pick an exact best response every step
    priors, _ = build_fictitious_play_priors(game, n_random=2, seed=0)
    for rule in TIE_RULES:
        history = run_fictitious_play(game, priors.clone(), rule, 40, [40], 0, logger)
        chosen = history["current_action_player_1"][-1]
        assert chosen.shape[0] == priors.shape[1]

    # Smoothed play: valid distributions respecting the exploration floor
    scores, _ = build_smoothed_scores(game, n_random=2, seed=0)
    history = run_smoothed_play(game, scores.clone(), (1 / 3, 1 / 4), 40, [40], logger)
    policy = history["final_policy"]
    assert bool((policy >= 0).all()) and bool(torch.isfinite(policy).all())
    assert bool(torch.allclose(policy.sum(-1), torch.ones_like(policy.sum(-1))))

    # Exp3 score-ratio identity, which is what makes the weights readable
    maker = AgentExp3MeanBased(8, arms, reward_range=(-0.875, 0.5))
    for _ in range(50):
        maker.act()
        maker.update(torch.rand(8) * 0.6 - 0.3)
    weights, eta = maker.weights.to(torch.float64), maker.get_learning_rate()
    exploitation = torch.softmax(eta * weights, dim=-1)
    ratio = torch.log(exploitation[:, 0] / exploitation[:, 1])
    predicted = eta * (weights[:, 0] - weights[:, 1])
    assert (ratio - predicted).abs().max().item() < 1e-06

    logger.info("  self test: all checks passed")
    return


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise(values) -> float:
    """Median of a batched diagnostic, or the value itself when scalar."""
    if isinstance(values, torch.Tensor):
        return float(values.double().median()) if values.numel() else float("nan")
    return float(values)


def render_grid_report(game: dict, results: dict) -> str:
    """Markdown report for one tick size."""
    arms, error = game["arms"], game["error"]
    benchmark = game["benchmark"]

    def quote(index):
        return f"({get_bid(arms[index]).item():.4f}, {get_ask(arms[index]).item():.4f})"

    lines = [
        f"# Falsification tests, delta = {game['delta']}",
        "",
        f"{game['n_arms']} arms. Benchmark a* {benchmark['a_star']:.4f}, "
        f"b* {benchmark['b_star']:.4f}, m* {benchmark['m_star']:.4f}.",
        "",
        "Special quotes, as (bid, ask) with the market error of the profile against itself:",
        "",
        "| quote | index | (bid, ask) | D against itself |",
        "|---|---|---|---|",
    ]
    for name, index in game["special"].items():
        if index is None:
            lines.append(f"| {name} | absent | — | — |")
        else:
            lines.append(
                f"| {name} | {index} | {quote(index)} | {error[index, index]:.4f} |"
            )

    lines += [
        "",
        "## Pure Nash equilibria",
        "",
        "| action | (bid, ask) | payoff | market error |",
        "|---|---|---|---|",
    ]
    for action in game["nash_actions"]:
        lines.append(
            f"| {action} | {quote(action)} | {game['payoff'][action, action]:.5f} | "
            f"{error[action, action]:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Rationalizable set: {game['layers'][-1]}, "
        f"{'equal to' if sorted(game['layers'][-1]) == game['nash_actions'] else 'different from'}"
        " the Nash actions."
    )

    if "A" in results:
        graph = results["A"]
        lines += [
            "",
            "## Test A: pure best-response graph",
            "",
            f"- components: {graph['n_components']}",
            f"- sink components: {len(graph['sink_components'])}",
            f"- self loops: {len(graph['self_loops'])}",
            f"- **sinks without a Nash action: {len(graph['sinks_without_nash'])}**",
            "",
            "| sink | size | actions | contains Nash | all Nash | max pairwise D |",
            "|---|---|---|---|---|---|",
        ]
        for number, component in enumerate(graph["sink_components"]):
            quotes = ", ".join(
                f"({b:.3f}, {a:.3f})" for b, a in component["quotes"][:4]
            )
            more = " ..." if component["size"] > 4 else ""
            lines.append(
                f"| {number} | {component['size']} | {quotes}{more} | "
                f"{component['contains_nash_action']} | "
                f"{component['all_actions_are_nash_actions']} | "
                f"{component['max_pairwise_market_error']:.4f} |"
            )

    if "B" in results:
        mixed = results["B"]
        lines += [
            "",
            "## Test B: mixed Nash equilibria",
            "",
            f"Searched symmetric equilibria up to support "
            f"{mixed['searched_max_support']}"
            f"{' (exhaustive)' if mixed['exhaustive'] else ' (bounded, so a null result is not proof of absence)'}.",
            "",
            "| support | probabilities | exploitability | E[D] | bad 0.05 | bad 0.10 | bad 0.20 |",
            "|---|---|---|---|---|---|---|",
        ]
        for record in mixed["equilibria"]:
            probabilities = ", ".join(f"{p:.3f}" for p in record["probabilities"])
            lines.append(
                f"| {record['support_player_1']} | {probabilities} | "
                f"{record['exploitability']:.2e} | {record['expected_market_error']:.4f} | "
                f"{record['bad_market_mass_eps_050']:.3f} | "
                f"{record['bad_market_mass_eps_100']:.3f} | "
                f"{record['bad_market_mass_eps_200']:.3f} |"
            )
        lines.append("")
        lines.append(
            f"Market error over equilibria: min {mixed['min_expected_market_error']:.4f}, "
            f"max {mixed['max_expected_market_error']:.4f}, against delta {game['delta']}."
        )

    for key, title in (
        ("C", "Test C: exact fictitious play"),
        ("D", "Test D: deterministic smoothed fictitious play"),
        ("E", "Test E: stochastic AgentExp3MeanBased"),
    ):
        if key not in results:
            continue
        lines += ["", f"## {title}", ""]
        for label, history in results[key].items():
            final = {
                name: summarise(values[-1])
                for name, values in history.items()
                if isinstance(values, list) and values
            }
            tail = history["tail_mean_market_error"]
            pieces = (
                [f"E[D] {final.get('expected_market_error', float('nan')):.4f}"]
                if key != "C"
                else [
                    f"current D {final.get('current_market_error', float('nan')):.4f}",
                    f"Cesaro D {final.get('cesaro_market_error', float('nan')):.4f}",
                ]
            )
            pieces.append(
                f"exploitability {final.get('product_exploitability', float('nan')):.2e}"
            )
            pieces.append(f"tail10 D {summarise(tail['10']):.4f}")
            lines.append(f"- `{label}`: " + ", ".join(pieces))

    return "\n".join(lines) + "\n"


def write_summary_rows(rows: list[dict]) -> str:
    """Master table as CSV, with NaN where a diagnostic does not apply."""
    lines = [",".join(SUMMARY_FIELDS)]
    for row in rows:
        values = []
        for field in SUMMARY_FIELDS:
            value = row.get(field, float("nan"))
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def plot_history(histories: dict, keys: tuple, title: str, ylabel: str, log_y: bool):
    """One diagnostic against the round, on a logarithmic horizontal axis."""
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")

    for colour, (name, history) in zip(PALETTE, histories.items()):
        rounds = history["round"]
        for style, key in zip(("-", "--", ":"), keys):
            if key not in history:
                continue
            values = torch.stack(
                [
                    (
                        v.double().median().reshape(())
                        if isinstance(v, torch.Tensor)
                        else torch.tensor(float(v))
                    )
                    for v in history[key]
                ]
            )
            axis.plot(
                rounds,
                values,
                color=colour,
                linestyle=style,
                linewidth=2,
                label=f"{name} {key}" if len(keys) > 1 else name,
            )

    axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
    axis.set_xlabel("Round")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend(fontsize=8, frameon=False)
    return figure


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(args) -> None:
    manager = ResultsManager(args.results_dir)
    tests = [t.strip().upper() for t in args.tests.split(",") if t.strip()]

    with manager.new_experiment(name="exp3_falsification") as exp:
        logger = setup_logger(
            log_path=exp.file("execution.log"), capture_loggers=["lama_lab"]
        )

        try:
            torch.manual_seed(args.seed)
            samples = UniformGenerator(0.0, 1.0).generate(args.n_samples)

            if args.self_test:
                logger.info("Running self tests.")
                self_test(samples, logger)
                if not tests:
                    return

            checkpoints = build_log_checkpoints(args.rounds)
            artifacts, rows, grid_tails = {}, [], []

            for delta in args.deltas:
                game = build_game(delta, samples)
                logger.info(
                    f"delta={delta}  arms={game['n_arms']}  "
                    f"pure Nash={len(game['nash_actions'])}  "
                    f"rationalizable={len(game['layers'][-1])}"
                )
                results = {}

                if "A" in tests:
                    results["A"] = test_best_response_graph(game, logger)

                if "B" in tests:
                    cap = (
                        None
                        if game["n_arms"] <= args.exhaustive_arms
                        else args.max_support
                    )
                    results["B"] = test_mixed_nash(game, cap, logger)

                if "C" in tests:
                    priors, labels = build_fictitious_play_priors(
                        game, args.n_random, args.seed
                    )
                    results["C"] = {}
                    for rule in TIE_RULES:
                        history = run_fictitious_play(
                            game,
                            priors.clone(),
                            rule,
                            args.rounds,
                            checkpoints,
                            args.seed,
                            logger,
                        )
                        results["C"][rule] = history
                        rows.extend(
                            collect_rows(
                                game,
                                history,
                                labels,
                                "fictitious_play",
                                "exact",
                                rule,
                                args.rounds,
                            )
                        )

                if "D" in tests:
                    scores, labels = build_smoothed_scores(
                        game, args.n_random, args.seed
                    )
                    results["D"] = {}
                    for name in args.schedules:
                        history = run_smoothed_play(
                            game,
                            scores.clone(),
                            SCHEDULES[name],
                            args.rounds,
                            checkpoints,
                            logger,
                        )
                        results["D"][name] = history
                        rows.extend(
                            collect_rows(
                                game,
                                history,
                                labels,
                                "smoothed_fictitious_play",
                                name,
                                "none",
                                args.rounds,
                            )
                        )

                if "E" in tests:
                    results["E"] = {}
                    for name in args.schedules:
                        history = run_exp3(
                            game,
                            SCHEDULES[name],
                            args.seeds,
                            args.rounds,
                            checkpoints,
                            args.seed,
                            logger,
                        )
                        results["E"][name] = history
                        rows.extend(
                            collect_rows(
                                game,
                                history,
                                [f"seed_{i}" for i in range(args.seeds)],
                                "exp3_mean_based",
                                name,
                                "none",
                                args.rounds,
                            )
                        )
                        grid_tails.append(
                            (
                                delta,
                                name,
                                summarise(history["tail_mean_market_error"]["10"]),
                            )
                        )

                tag = f"delta_{delta}".replace(".", "p")
                artifacts[f"report_{tag}"] = render_grid_report(game, results)
                artifacts[f"payoff_{tag}"] = game["payoff"].cpu()
                artifacts[f"market_error_{tag}"] = game["error"].cpu()
                artifacts[f"structure_{tag}"] = {
                    "delta": delta,
                    "n_arms": game["n_arms"],
                    "benchmark": game["benchmark"],
                    "special": game["special"],
                    "nash_actions": game["nash_actions"],
                    "rationalizable": game["layers"][-1],
                    "best_response_graph": results.get("A"),
                    "mixed_nash": results.get("B"),
                }

                for key, series, ylabel in (
                    ("D", ("expected_market_error",), "E[D]"),
                    ("D", ("bad_0.1",), "P(D > 0.1)"),
                    ("D", ("product_exploitability",), "Exploitability"),
                    ("E", ("expected_market_error",), "E[D]"),
                    ("E", ("bad_0.1",), "P(D > 0.1)"),
                    ("E", ("product_exploitability",), "Exploitability"),
                ):
                    if key in results:
                        artifacts[f"plot_{key}_{series[0]}_{tag}"] = plot_history(
                            results[key], series, f"{key}, delta {delta}", ylabel, False
                        )

            artifacts["master_summary"] = write_summary_rows(rows)
            if grid_tails:
                figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
                for colour, name in zip(PALETTE, args.schedules):
                    points = [(d, v) for d, s, v in grid_tails if s == name]
                    if points:
                        axis.plot(
                            *zip(*points), "o-", color=colour, linewidth=2, label=name
                        )
                axis.plot(
                    [min(args.deltas), max(args.deltas)],
                    [min(args.deltas), max(args.deltas)],
                    color=PALETTE[3],
                    linestyle="--",
                    label="market error = delta",
                )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_xlabel("delta")
                axis.set_ylabel("Tail market error")
                axis.set_title("Tail market error against tick size", fontsize=11)
                axis.grid(True, linestyle="--", alpha=0.7)
                axis.legend(fontsize=9, frameon=False)
                artifacts["plot_grid_summary"] = figure

            exp.save_all(artifacts)
            plt.close("all")
            logger.info(f"Saved to {exp.path}.")

            for delta in args.deltas:
                print(artifacts[f"report_delta_{delta}".replace(".", "p")])

        except Exception:
            logger.error(traceback.format_exc())
            plt.close("all")
            raise
    return


def collect_rows(
    game, history, labels, algorithm, schedule, rule, n_rounds
) -> list[dict]:
    """One master-table row per run inside a batched history."""

    def last(name, index):
        values = history.get(name)
        if not isinstance(values, list) or not values:
            return float("nan")
        value = values[-1]
        if isinstance(value, torch.Tensor):
            return float(value.double().reshape(-1)[index])
        return float(value)

    rows = []
    for index, label in enumerate(labels):
        row = {
            "algorithm": algorithm,
            "schedule": schedule,
            "delta": game["delta"],
            "K": game["n_arms"],
            "seed": index,
            "initialization": label,
            "tie_break_rule": rule,
            "T_final": n_rounds,
            "final_current_expected_market_error": last("expected_market_error", index),
            "final_bad_market_mass_eps_005": last("bad_0.05", index),
            "final_bad_market_mass_eps_010": last("bad_0.1", index),
            "final_bad_market_mass_eps_020": last("bad_0.2", index),
            "cesaro_market_error": last("cesaro_market_error", index),
            "fraction_bad_market_eps_005": last("fraction_bad_0.05", index),
            "fraction_bad_market_eps_010": last("fraction_bad_0.1", index),
            "fraction_bad_market_eps_020": last("fraction_bad_0.2", index),
            "tail_mean_market_error": float(
                history["tail_mean_market_error"]["10"].double().reshape(-1)[index]
            ),
            "tail_bad_market_mass_eps_010": last("bad_0.1", index),
            "tail_product_exploitability": last("product_exploitability", index),
            "product_exploitability": last("product_exploitability", index),
            "max_external_regret": last("max_avg_regret", index),
            "independence_TV": last("independence_tv", index),
            "policy_L1_distance": last("policy_l1_distance", index),
            "support_br_gap": last("support_br_gap_1", index),
            "worst_active_gap_tau_1e3": last("worst_active_gap_tau_1e-3_1", index),
            "worst_active_gap_tau_1e4": last("worst_active_gap_tau_1e-4_1", index),
            "argmax_action_player_1": last("argmax_1", index),
            "argmax_action_player_2": last("argmax_2", index),
            "max_probability_player_1": last("max_probability_1", index),
            "max_probability_player_2": last("max_probability_2", index),
        }
        for name in ("q_star", "ask_special", "bid_special", "withdrawn"):
            row[f"p1_{name}"] = last(f"p1_{name}", index)
            row[f"p2_{name}"] = last(f"p2_{name}", index)
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lama Lab - Exp3 fictitious-play falsification tests"
    )
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--tests", type=str, default="A,B,C,D,E")
    parser.add_argument(
        "--deltas", type=float, nargs="+", default=[0.125, 0.0625, 0.05]
    )
    parser.add_argument("--rounds", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--max_support", type=int, default=3)
    parser.add_argument("--exhaustive_arms", type=int, default=12)
    parser.add_argument(
        "--schedules",
        type=str,
        nargs="+",
        default=["theory_schedule"],
        choices=list(SCHEDULES),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    main(parser.parse_args())
