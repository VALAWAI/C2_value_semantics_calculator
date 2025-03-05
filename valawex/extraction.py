from copy import deepcopy
from dataclasses import dataclass
import logging
import random

from sympy import init_printing, Add, Mul, Pow, Expr, oo, zoo, nan, latex
from valawex.analysis import U_IN, U_OUT, Game, Patient, crit_symbols, action_symbols, postcrit_symbols

from typing import List, Callable, Any, Tuple

logging.basicConfig()

init_printing()
random.seed(100)

def Subs(a, b):
    return Add(a, Mul(b, -1))

def Div(a, b):
    return Mul(a, Pow(b, -1))

OPER = [Add, Subs, Mul, Div]
RAND_INT_RANGE = 10


def generate_child_expression(expr: Expr) -> Expr:
    """Expand an expression by adding an operation.

    Parameters
    ----------
    expr : Expr

    Returns
    -------
    Expr
    """
    dice = random.randint(0, 3)
    if dice == 0:
        next_arg = crit_symbols[2]
    elif dice == 1:
        next_arg = postcrit_symbols[1]
    elif dice == 2:
        next_arg = action_symbols[0]
    else:
        next_arg = random.randint(-RAND_INT_RANGE, RAND_INT_RANGE)

    next_op = random.choice(OPER)
    two_coin = random.random()
    if two_coin < 0.5:
        return next_op(expr, next_arg)
    else:
        return next_op(next_arg, expr)


def generate_child_expression_OLD(expr: Expr) -> Expr:
    """Expand an expression by adding an operation.

    Parameters
    ----------
    expr : Expr

    Returns
    -------
    Expr
    """
    three_coin = random.random()
    if three_coin < 1/3:
        next_arg = U_IN
    elif three_coin < 2/3:
        next_arg = U_OUT
    else:
        next_arg = random.randint(-RAND_INT_RANGE, RAND_INT_RANGE)

    next_op = random.choice(OPER)
    two_coin = random.random()
    if two_coin < 0.5:
        return next_op(expr, next_arg)
    else:
        return next_op(next_arg, expr)



def generate_valid_child(expr: Expr) -> Expr:
    """Generate a valid child.

    A valid child expression is one which is not constant, i.e. not all
    variables have been removed and does not have imaginary constants.

    Parameters
    ----------
    expr : Expr

    Returns
    -------
    Expr
    """
    child = generate_child_expression(expr)
    while len(child.free_symbols) == 0 or child.has(oo, -oo, zoo, nan):
        child = generate_child_expression(expr)
    return child


def compute_affinity_OLD(
    games: List[Game],
    choices: List[int],
    expr: Expr,
    method: str,
    **kwargs
) -> float:
    """Affinity of a set of empirical choices for a behaviour expression.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
    expr : Expr
    method : str
        The evaluation method. Must be one of ``ideal_choice_set``,
        ``continuous_evaluation`` or ``equidistant_ranks``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``games`` and ``choices`` have different lengths.
    """
    if len(games) != len(choices):
        raise ValueError("game list and choices list must have the same length")
    
    num = 0
    den = 0
    for g, c in zip(games, choices):
        print("G", g, method)
        func = getattr(g, method)
        res = func(expr, **kwargs)

        if not res:
            continue
        else:
            den += 1

        if method == 'ideal_choice_set':
            if c in res:
                num += 1
        else:

            num += res[c]


    return num / den


def compute_affinity(
        patients: List[Patient],
        choices: List[int],
        expr: Expr,
        method: str,
        **kwargs
) -> float:
    """Affinity of a set of empirical choices for a behaviour expression.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
    expr : Expr
    method : str
        The evaluation method. Must be one of ``ideal_choice_set``,
        ``continuous_evaluation`` or ``equidistant_ranks``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``games`` and ``choices`` have different lengths.
    """

    method = 'alignment_evaluation'

    num = 0
    den = 0
    for p in patients:

        # We evaluate the patient according to the alignment function
        func = getattr(p, method)
        res = func(expr, **kwargs)

        quadratic_error = (res - p.get_alignment())**2

        den += 1
        num += -quadratic_error # we are optimising so we put the error as negative

    return num / den


def compute_fitness(
    games: List[Game],
    choices: List[int],
    expr: Expr,
    method: str,
    penal_function: Callable[[Expr], float] = lambda x: x.count_ops(),
    lamb: float = 0.00,
    **kwargs
) -> float:
    """Compute the fitness of an expression given a set of choices in games.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
    expr : Expr method : str
        The evaluation method. Must be one of ``ideal_choice_set``,
        ``continuous_evaluation`` or ``equidistant_ranks``.
    penal_function : Callable[[Expr], float], optional
        Complexity penalization, by default the number of operations in the
        expression.
    lamb : float, optional
        Weight of the complexity penalization, by default 0.01.
    **kwargs
        Keyword arguments to compute the affinity.

    Returns
    -------
    float
    """
    aff = compute_affinity(games, choices, expr, method, **kwargs)
    pen = penal_function(expr)
    return aff - lamb*pen


@dataclass
class Candidate:
    """A candidate in the affinity-maximizing search."""
    expr: Expr
    fitness: float = None


def evolution_strategy(
    rid: Any,
    games: List[Game],
    choices: List[int],
    method: str,
    mu: int,
    lam: int,
    penal_function: Callable[[Expr], float] = lambda x: x.count_ops(),
    lamb: float = 0.00,
    max_partial_iters: int = 10,
    max_total_iters: int = 100,
    threshold: float = 0.99,
    **kwargs
) -> Tuple[Any, Expr, float]:
    """Simple evolutionary search strategy.

    Parameters
    ----------
    rid : Any
        An identifier for the optimization run.
    games : List[Game]
    choices : List[int]
    method : str
        The evaluation method. Must be one of ``ideal_choice_set``,
        ``continuous_evaluation`` or ``equidistant_ranks``.
    mu : int
        Number of the best candidates that survive from one generation to the
        next.
    lam : int
        Number of total children generated at every iteration.
    penal_function : Callable[[Expr], float], optional
        Complexity penalization, by default the number of operations in the
        expression.
    lamb : float, optional
        Weight of the complexity penalization, by default 0.01.
    max_partial_iters : int, optional
        Maximum number of interations without improvement before halting, by
        default 20.
    max_total_iters : int, optional
        Maximum number of iterations before halting regardless of improvement,
        by default 100.
    threshold : float, optional
        If a candidate surpassing this fitness in encountered, the search is
        automatically halted, by default 0.95.
    
    Returns
    -------
    Tuple[Expr, float]
        The optimal expression that fits the set of choices in the set of games,
        and its affinity.

    Raises
    ------
    ValueError
        If ``mu`` or ``lam`` are negative.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if mu < 0 or lam < 0:
        raise ValueError("mu or lam cannot be negative")

    # build initial population
    pop = list()

    for c in crit_symbols:
        pop.append(Candidate(c))

    for a in action_symbols:
        pop.append(Candidate(a))

    for pc in postcrit_symbols:
        pop.append(Candidate(pc))


    for cand in pop:
        cand.fitness = compute_fitness(games, choices, cand.expr, method,
            penal_function=penal_function, lamb=lamb, **kwargs)
    pop.sort(key=lambda c: c.fitness, reverse=True)
    best_so_far = deepcopy(pop[0].expr)
    max_fitness = pop[0].fitness

    logger.info(f"Initial population: {', '.join([str(c.expr) for c in pop])}")
    logger.info(f"Best candidate: {best_so_far}")
    logger.info(f"Fitness: {max_fitness:.4f}\n")

    partial_iters = 0
    total_iters = 0
    while (partial_iters < max_partial_iters) and \
        (total_iters < max_total_iters) and \
        (max_fitness < threshold):

        # prepare parents
        if mu > len(pop):
            parents = pop
        else:
            parents = pop[:mu]

        # prepare next generation
        children_per_parent = lam // len(parents)
        children = []
        for par in parents:
            for _ in range(children_per_parent):
                child_expr = generate_valid_child(par.expr)
                child_fitness = compute_fitness(games, choices, child_expr,
                    method, penal_function=penal_function, lamb=lamb, **kwargs)
                children.append(Candidate(child_expr, child_fitness))
        pop = children + parents
        pop.sort(key=lambda c: c.fitness, reverse=True)

        # assess the progress made in this iteration
        logger.info(f"ITERATION {total_iters}")
        if pop[0].fitness <= max_fitness:
            partial_iters += 1
            logger.info("No new best candidate found\n")
        else:
            best_so_far = deepcopy(pop[0].expr)
            max_fitness = pop[0].fitness
            partial_iters = 0
            logger.info(f"New best candidate: {best_so_far}")
            logger.info(f"Fitness: {max_fitness:.4f}\n")
        total_iters += 1

    return rid, best_so_far, max_fitness


def evolution_strategy_OLD(
        rid: Any,
        games: List[Game],
        choices: List[int],
        method: str,
        mu: int,
        lam: int,
        penal_function: Callable[[Expr], float] = lambda x: x.count_ops(),
        lamb: float = 0.00,
        max_partial_iters: int = 20,
        max_total_iters: int = 100,
        threshold: float = 0.95,
        **kwargs
) -> Tuple[Expr, float]:
    """Simple evolutionary search strategy.

    Parameters
    ----------
    rid : Any
        An identifier for the optimization run.
    games : List[Game]
    choices : List[int]
    method : str
        The evaluation method. Must be one of ``ideal_choice_set``,
        ``continuous_evaluation`` or ``equidistant_ranks``.
    mu : int
        Number of the best candidates that survive from one generation to the
        next.
    lam : int
        Number of total children generated at every iteration.
    penal_function : Callable[[Expr], float], optional
        Complexity penalization, by default the number of operations in the
        expression.
    lamb : float, optional
        Weight of the complexity penalization, by default 0.01.
    max_partial_iters : int, optional
        Maximum number of interations without improvement before halting, by
        default 20.
    max_total_iters : int, optional
        Maximum number of iterations before halting regardless of improvement,
        by default 100.
    threshold : float, optional
        If a candidate surpassing this fitness in encountered, the search is
        automatically halted, by default 0.95.

    Returns
    -------
    Tuple[Expr, float]
        The optimal expression that fits the set of choices in the set of games,
        and its affinity.

    Raises
    ------
    ValueError
        If ``mu`` or ``lam`` are negative.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if mu < 0 or lam < 0:
        raise ValueError("mu or lam cannot be negative")

    # build initial population
    pop = [Candidate(U_IN), Candidate(U_OUT)]
    for cand in pop:
        cand.fitness = compute_fitness(games, choices, cand.expr, method,
                                       penal_function=penal_function, lamb=lamb, **kwargs)
    pop.sort(key=lambda c: c.fitness, reverse=True)
    best_so_far = deepcopy(pop[0].expr)
    max_fitness = pop[0].fitness

    logger.info(f"Initial population: {', '.join([str(c.expr) for c in pop])}")
    logger.info(f"Best candidate: {best_so_far}")
    logger.info(f"Fitness: {max_fitness:.4f}\n")

    partial_iters = 0
    total_iters = 0
    while (partial_iters < max_partial_iters) and \
            (total_iters < max_total_iters) and \
            (max_fitness < threshold):

        # prepare parents
        if mu > len(pop):
            parents = pop
        else:
            parents = pop[:mu]

        # prepare next generation
        children_per_parent = lam // len(parents)
        children = []
        for par in parents:
            for _ in range(children_per_parent):
                child_expr = generate_valid_child(par.expr)
                child_fitness = compute_fitness(games, choices, child_expr,
                                                method, penal_function=penal_function, lamb=lamb, **kwargs)
                children.append(Candidate(child_expr, child_fitness))
        pop = children + parents
        pop.sort(key=lambda c: c.fitness, reverse=True)

        # assess the progress made in this iteration
        logger.info(f"ITERATION {total_iters}")
        if pop[0].fitness <= max_fitness:
            partial_iters += 1
            logger.info("No new best candidate found\n")
        else:
            best_so_far = deepcopy(pop[0].expr)
            max_fitness = pop[0].fitness
            partial_iters = 0
            logger.info(f"New best candidate: {best_so_far}")
            logger.info(f"Fitness: {max_fitness:.4f}\n")
        total_iters += 1

    return rid, best_so_far, max_fitness


if __name__ == '__main__':



    patients = []

    for _ in range(150):
        patients.append(Patient())

    no_choices = None



    _, expr, max_fitness = evolution_strategy(
        None,
        patients,
        no_choices,
        'alignment_evaluation',
        50,
        1000,
        max_partial_iters=20,
        max_total_iters=100
    )

    print(expr)
    print("----")
    print(latex(expr))
    print("----")
    print(max_fitness)


"""
    # fine-tune the search for demo
    kwargs = {
        "max_partial_iters": 20,
        "fit_kwargs": {"lamb": 0.}
    }

    games = [
        Game.from_csv(f"example_games/matrix{i}.csv", delimiter=';', \
                      game_id=f"Game {i}")
        for i in range(1, 5)
    ]

    _ = [g.assign_groups('in', 'out') for g in games]

    random_choices = [random.randint(0, g.num_choices()-1) for g in games]

    # fine-tune the search for demo
    kwargs = {
        "max_partial_iters": 25,
        "fit_kwargs": {"lamb": 0.}
    }

    _, expr, max_fitness = evolution_strategy(
        None,
        games,
        random_choices,
        'continuous_evaluation',
        5,
        100,
        max_partial_iters=40,
        max_total_iters=200
    )

    print(expr)
    print("----")
    print(latex(expr))
    print("----")
    print(max_fitness)







"""