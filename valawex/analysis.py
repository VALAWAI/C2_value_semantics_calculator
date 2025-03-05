from collections import OrderedDict
from copy import deepcopy
import csv
import json
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Pow, Expr, Piecewise, Gt, Eq, init_printing
from typing import Any, Dict, List, Tuple, Type, TypeVar

G = TypeVar('G', bound='Game')




crit_symbols  = symbols('c(1:17)')
action_symbols  = symbols('a(1:12)')
postcrit_symbols  = symbols('pc(1:17)')



init_printing()


class Choice:
    """Class for a choice in a game.

    Examples
    --------
    >>> c = Choice(2, 5)
    >>> c['top']
    2
    >>> c['bottom']
    5
    >>> print(c)
    (2, 5)
    """
    __top: int
    __bottom: int

    def __init__(self, top: int, bottom: int):
        self.__top = top
        self.__bottom = bottom

    def __getitem__(self, index: str) -> int:
        if index != 'top' and index != 'bottom':
            raise IndexError("Choice indexing must be either 'top' or 'bottom'")
        if index == 'top':
            return self.__top
        elif index == 'bottom':
            return self.__bottom

    def __str__(self) -> str:
        return f"({self.__top}, {self.__bottom})"


class Patient:

    def __init__(self, criteria: List[float] = [], actions: List[bool] = [], post_criteria: List[float] = []):
        self.__criteria = []
        self.__actions = []
        self.__post_criteria = []
        self.__alignment = None


        for c in criteria:
            self.__criteria.append(c)
        for a in actions:
            self.__actions.append(a)
        for pc in post_criteria:
            self.__post_criteria.append(pc)

        if self.num_criteria() == 0:
            self.__criteria = np.random.rand(16).tolist()
        if self.num_post_criteria() == 0:
            self.__post_criteria = np.random.rand(16).tolist()
        if self.num_actions() == 0:
            self.__actions = np.zeros(11).tolist()
            chosen_action = np.random.randint(11)
            self.__actions[chosen_action] = 1

        self.set_alignment()

    def get_criteria(self) -> List[float]:
        return self.__criteria

    def get_actions(self) -> List[bool]:
        return self.__actions

    def get_post_criteria(self) -> List[float]:
        return self.__post_criteria

    def get_features(self) -> List[float]:

        return self.__criteria + self.__actions + self.__post_criteria

    def get_alignment(self) -> float:
        return self.__alignment


    def num_criteria(self) -> int:
        """Get the number of choices in the game.

        Returns
        -------
        int
        """
        return len(self.__criteria)

    def num_post_criteria(self) -> int:
        """Get the number of choices in the game.

        Returns
        -------
        int
        """
        return len(self.__post_criteria)

    def num_actions(self) -> int:
        """Get the number of choices in the game.

        Returns
        -------
        int
        """
        return len(self.__actions)


    def set_alignment(self) -> None:

        "Placeholder until we have real data"
        
        self.__alignment = 2*self.__post_criteria[1] +  7*self.__actions[7]

    def alignment_evaluation(self, function: Expr) -> float:
        """Evaluate the criteria according to all four values
        """

        crit = self.get_criteria()
        act = self.get_actions()
        post_crit = self.get_post_criteria()

        treatment = dict()

        for i in range(len(crit)):
            treatment[crit_symbols[i]] = crit[i]
            treatment[postcrit_symbols[i]] = post_crit[i]

        for i in range(len(act)):
            treatment[action_symbols[i]] = act[i]

        #print("Yes, this line 121 in analysis.py is executed!")
        #print("Expression : ", function)
        try:
            evaluation = function.evalf(subs=treatment)
        except ZeroDivisionError:
            evaluation = 999999999

        return evaluation


class Game:
    """Class for a game as a set of possible choices.

    Examples
    --------
    >>> g = Game(Choice(2, 3), Choice(7, 1), game_id='my_game')
    >>> print(g)
    my_game:
    (2, 3)
    (7, 1)

    >>> print(g[0])
    (2, 3)
    >>> print(g[1])
    (7, 1)
    >>> [print(i) for i in g.get_choices()]
    (2, 3)
    (7, 1)
    [None, None]
    >>> g.num_choices()
    2
    """
    __choices: List[Choice]
    __groups: Dict[str, str]

    def __init__(self, *choices: Choice, game_id: Any = None) -> None:
        self.__choices = []
        for c in choices:
            self.__choices.append(c)
        self.id = game_id

    def __getitem__(self, item: int) -> Choice:
        return self.__choices[item]

    def __str__(self) -> str:
        newline = '\n'
        choices_str = newline.join(str(c) for c in self.__choices)
        if self.id:
            return f"{self.id}:\n{choices_str}"
        else:
            return f"Game:\n{choices_str}"

    def assign_groups(self, top_group: str, bottom_group: str) -> None:
        """Assign the tokens in the top or bottom to the in- and/or out-group.

        Parameters
        ----------
        top_group : str
        bottom_group : str

        Raises
        ------
        ValueError
            If ``top_group`` or ``bottom_group`` is not ``in`` or ``out``.
        """
        options = ('in', 'out')
        if not top_group in options or not bottom_group in options:
            raise ValueError("Choices must be assigned to either 'in' or 'out'")
        self.__groups = {'top': top_group, 'bottom': bottom_group}

    def get_groups(self) -> Dict[str, str]:
        return self.__groups

    @classmethod
    def from_csv(
        cls: Type[G],
        filename: str,
        top_col: int = 0,
        bottom_col: int = 1,
        game_id: Any = None,
        **kwargs
    ) -> G:
        """Parse a game from a csv file.

        The csv file must contain one row per choice if the game. Which column
        represents the points at the top or bottom of the chip is customizable.

        Parameters
        ----------
        filename : str
            The csv file.
        top_col : int, optional
            The column index containing the points at the top of the chip, by
            default 0.
        bottom_col : int, optional
            The column index containing the points at the bottom of the chip, by
            default 1.
        game_id : Any, optional
        **kwargs
            Additional arguments for csv parsing.

        Returns
        -------
        G : Game
            A game with the set of choices parsed form the csv file.

        Raises
        ------
        ValueError
            If top_col and bottom_col are equal.

        See Also
        --------
        csv.reader
        """
        if top_col == bottom_col:
            raise ValueError("in_col and out_col must be different")
        choices = []
        with open(filename, 'r') as game_file:
            reader = csv.reader(game_file, **kwargs)
            for row in reader:
                c = Choice(int(row[top_col]), int(row[bottom_col]))
                choices.append(c)
        game_file.close()
        return cls(*choices, game_id=game_id)

    def get_choices(self) -> List[Choice]:
        """Get the choices in the game as a list.

        Returns
        -------
        List[Choice]
        """
        return self.__choices

    def num_choices(self) -> int:
        """Get the number of choices in the game.

        Returns
        -------
        int
        """
        return len(self.__choices)
    
    def get_choice_index(self, top: int, bottom: int) -> int:
        """Get the index of a choice in the choice list of the game.

        The choice is not passed as a Choice object, but as the tuple of in-
        and out-group tokens.

        Parameters
        ----------
        top : int
        bottom : int

        Returns
        -------
        int

        Raises
        ------
        LookupError
            If the choice is not found.
        """
        for i, c in enumerate(self.__choices):
            if c['top'] == top and c['bottom'] == bottom:
                return i
        raise LookupError("Game does not contain choice " +
                          f"({top, bottom}).")



    def __in_out_groups(self) -> Tuple[str, str]:
        if self.__groups['top'] == self.__groups['bottom']:
            raise TypeError("""the bottom and top choices must be assigned to \
                different groups""")
        if self.__groups['top'] == 'in' and self.__groups['bottom'] == 'out':
            ingroup = 'top'
            outgroup = 'bottom'
        else:
            ingroup = 'bottom'
            outgroup = 'top'
        return ingroup, outgroup
    
    def ideal_choice_set(self, function: Expr, tol: float = 1.E-5) \
        -> List[int]:
        """Compute the ideal choice set for some pre-defined profile.

        Parameters
        ----------
        function : Expr
            Function that the profile seeks to maximize
        tol : float, optional
            When to consider an evaluation is a new maximum, by default 1.E-5

        Returns
        -------
        List[Choice]

        Raises
        ------
        ValueError
            if `tol` is negative
        """
        choice_set = []
        if tol < 0:
            raise ValueError("tolerance parameter must be positive")
        ingroup, outgroup = self.__in_out_groups()
        maximum = float('-inf')
        for i, c in enumerate(self.__choices):
            evaluation = function.evalf(subs={U_IN: c[ingroup], U_OUT: c[outgroup]})
            if evaluation-maximum > tol:
                maximum = evaluation
                choice_set = [i]
            elif abs(evaluation-maximum) < tol:
                choice_set.append(i)
        return choice_set
    
    def continuous_evaluation(self, function: Expr) -> List[float]:
        """Evaluate the choices continously according to some profile.

        The choices are assigned a score ranging from 0 (for the choice that is
        the least aligned with a profile) to 1 (for the choice that is the most
        aligned with a profile). Intermediate choices are assigned a score that
        is normalized by the range of the evaluations.

        Parameters
        ----------
        function : Expr

        Returns
        -------
        List[float]
            The order in the evaluations returned respects the order of the
            choices in the game.
        None
            If all the choices in the game are equally evaluated.
        """
        evaluations = [0] * self.num_choices()
        ingroup, outgroup = self.__in_out_groups()
        for i, c in enumerate(self.__choices):
            evaluations[i] = function.evalf(subs={U_IN: c[ingroup], U_OUT: c[outgroup]})
        minimum = min(evaluations)
        maximum = max(evaluations)
        if maximum == minimum:
            return None
        denominator = maximum - minimum
        for i, e in enumerate(evaluations):
            evaluations[i] = (e - minimum) / denominator
        return evaluations
    
    def equidistant_ranks(self, function: Expr) -> List[float]:
        """Rank the choices in the game according to some profile.

        The choices are assigned a rank from 0 (for the choice that is the least
        aligned with a profile) to 1 (for the choice that is the most aligned
        with a profile). Intermediate choices are assigned a score with *equal
        distancing* between pairs of consecutive choices.

        Parameters
        ----------
        function : Expr

        Returns
        -------
        List[float]
            The order in the ranks returned respects the order of the choices in
            the game.
        None
            If all the choices in the game are equally evaluated.
        """
        cont_evals = self.continuous_evaluation(function)
        if not cont_evals:
            return None
        old_index = {e: [] for e in cont_evals}
        for i, e in enumerate(cont_evals):
            old_index[e].append(i)

        unique_evals = list(set(cont_evals))
        unique_evals.sort()
        new_index = {e: i for i, e in enumerate(unique_evals)}

        denom = len(unique_evals) - 1
        ranks = [0] * self.num_choices()
        for e, old in old_index.items():
            for o in old:
                ranks[o] = new_index[e] / denom
        return ranks


U_IN, U_OUT = symbols('u_{in}, u_{out}')
"""Symbols for in- and out-group tokens arguments."""

EGALITARIAN = Piecewise(
    (Pow(U_IN - U_OUT, -1), Gt(U_IN, U_OUT)),
    (Pow(U_OUT - U_IN, -1), Gt(U_OUT, U_IN)),
    (100, Eq(U_IN, U_OUT))
)
"""Egalitarian profile minimizes difference between in- and out-group."""

DIFFERENTARIAN = Pow(EGALITARIAN, -1)
"""Differentarian profile maximizes difference between in- and out-group."""

SECTARIAN = U_IN
"""Sectarian profile maximizes in-group."""

SELFHATER = Pow(SECTARIAN, -1)
"""Self-hater profile minimizes in-group."""

BADPERSON = Pow(U_OUT, -1)
"""Bad person profile minimizes out-group."""

ALTRUIST = Pow(BADPERSON, -1)
"""Altruist profile maximizes out-group."""

UTILITARIAN = U_IN + U_OUT
"""Utilitarian profile maximizes total payoff."""

SOCIOPATH = Pow(UTILITARIAN, -1)
"""Sociopath profile minimizes total payoff."""

NUM_AXIS = 4
IDEAL_PROFILES = OrderedDict(
    [
        ('Egalitarian',     EGALITARIAN),
        ('Sectarian',       SECTARIAN),
        ('Altruist',        ALTRUIST),
        ('Utilitarian',     UTILITARIAN),
        ('Differentarian',  DIFFERENTARIAN),
        ('Self-hater',      SELFHATER),
        ('Bad person',      BADPERSON),
        ('Sociopath',       SOCIOPATH)
    ]
)

METHODS = ['ideal_choice_set', 'continuous_evaluation', 'equidistant_ranks']

def get_opposite_profile(prof: str) -> str:
    """Given an ideal profile, find its opposite.

    Parameters
    ----------
    prof : str

    Returns
    -------
    str
    """
    ordered_profs = list(IDEAL_PROFILES.keys())
    i = ordered_profs.index(prof)
    if i < NUM_AXIS:
        return ordered_profs[NUM_AXIS+i]
    else:
        return ordered_profs[i-NUM_AXIS]


def __evaluate_game(game: Game, method: str, **kwargs) -> Dict[str, List[Any]]:
    func = getattr(game, method)
    result = {
        pr: func(expr, **kwargs)
        for pr, expr in IDEAL_PROFILES.items()
    }
    return result
    

def compute_participant_profile(
    games: List[Game],
    choices: List[int],
    method: str,
    **kwargs
) -> Dict[str, float]:
    """Compute the profile of a participant against all the ideal profiles.

    If a game is not informative with respect to a particular profile, it is
    note considered when taking the average over the evaluation of all games.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
        Index of the choices selected at each game.
    method : str
        The method used to evaluate a choice by a profile in a given game.
        Options are: ``ideal_choice_set``, ``continuous_evaluation`` or
        ``equidistant_ranks``.

    Returns
    -------
    Dict[str, float]
        A map from each ideal profile to the affinity shown by the set of
        choices.

    Raises
    ------
    ValueError
        If the number of games and choices do not match.
    """
    num_games = len(games)
    if num_games != len(choices):
        raise ValueError("number of choices and number of games must be equal")
    
    attr_name = f"__{method}_computed"
    profile = {p: 0. for p in IDEAL_PROFILES.keys()}
    info_games = {p: 0. for p in IDEAL_PROFILES.keys()}
    for g, c in zip(games, choices):
        # evaluate each game only once according to the method of choice
        try:
            res = getattr(g, attr_name)
        except AttributeError:
            res = __evaluate_game(g, method, **kwargs)
            setattr(g, attr_name, res)
        for p in IDEAL_PROFILES.keys():

            # if the game is not informative (the metric is None), move on
            # else record an additional informative game
            if not res[p]:
                continue
            else:
                info_games[p] += 1

            if method == 'ideal_choice_set':
                if c in res[p]:
                    profile[p] += 1
            else:
                profile[p] += res[p][c]
    
    norm_profile = {}
    for p in IDEAL_PROFILES.keys():
        try:
            norm_profile[p] = profile[p] / info_games[p]
        except ZeroDivisionError:
            norm_profile[p] = 0
    return norm_profile
        

def compute_plot_participant_profile(
    games: List[Game],
    choices: List[int],
    method: str,
    filename: str,
    **kwargs
) -> Dict[str, float]:
    """Compute and plot a profile for a participant.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
    method : str
    filename : str
        Where to save the generated lollipop plot.

    Returns
    -------
    Dict[str, float]
        The profile for the participant.

    See Also
    --------
    compute_participant_profile
    """
    profile = compute_participant_profile(games, choices, method, **kwargs)
    plot = lollipop_plot(profile)
    plot.savefig(filename, dpi=400)
    return profile


def compute_profile_4d(
    games: List[Game],
    choices: List[int],
    method: str,
    pid: Any,
    **kwargs
) -> Dict[str, int]:
    """Compute and transform the profile along four dimensions.

    Parameters
    ----------
    games : List[Game]
    choices : List[int]
    method : str
    pid : Any
        An identifier for the participant.

    Returns
    -------
    Dict[str, int]
        The adherence to each profile is quantified from 0 to 100.
    """
    profile8 = compute_participant_profile(games, choices, method, **kwargs)
    profile4 = {}
    for i in range(NUM_AXIS):
        p = list(IDEAL_PROFILES.keys())[i]
        p_inv = get_opposite_profile(p)
        try:
            profile4[p] = (profile8[p]-profile8[p_inv]) / (profile8[p]+profile8[p_inv])
        except ZeroDivisionError:
            profile4[p] = 0.
    for k, v in profile4.items():
        profile4[k] = round((v + 1)/2 * 100)
    profile4['id'] = pid
    return profile4


# pre-process games for final exoeriment
def __parse_games(filename: str) -> Dict[str, Game]:
    with open(filename, 'r') as f:
        game_data = json.load(f)
    games = {}
    for gdata in game_data:
        game_id = gdata['game_id']
        choices = []
        num_choices_per_game = 13
        for i in range(1, num_choices_per_game+1):
            c = gdata['choices'][f'choice_{i}']
            choices.append(Choice(*c))
        games[game_id] = Game(*choices, game_id=game_id)
    return games

# TODO point to the right file
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
__GAMES_FILE = f'{dir_path}/resources/task_description.json'
__GAMES = __parse_games(__GAMES_FILE)


def analyze_participant(
    participant_data: Dict[str, str],
    method: str,
    **kwargs
) -> Dict[str, Any]:
    """Analyze raw data in the value awareness experiment.

    Parameters
    ----------
    participant_data : Dict[str, str]
        A dictionary describing the games the participant is presented with and
        the decision they make.
    method : str
        Use 'equidistant_ranks'.

    Returns
    -------
    Dict[str, Any]
        A dictionary describing the value profile of the participant.
    """
    pid = participant_data['id']
    games = []
    choices = []
    for k, v in participant_data.items():
        if not k.endswith('group'):
            continue

        # identify game and assign groups
        gid = k.split('_')[0]
        groups = v.split(',')
        if groups[0] == groups[1]:
            continue
        games.append(deepcopy(__GAMES[gid]))
        games[-1].assign_groups(groups[0][:-5], groups[1][:-5])

        # get the choice
        c = participant_data[f'{gid}_values'].split(',')
        choice_ind = __GAMES[gid].get_choice_index(int(c[0]), int(c[1]))
        choices.append(choice_ind)

    profile4d = compute_profile_4d(games, choices, method, pid, **kwargs)
    return profile4d


plt.rcParams.update({'font.size': 22})

def lollipop_plot(
    profile: Dict[str, float],
    profile_compare: Dict[str, float] = None,
    profile_name: str = None,
    profile_compare_name: str = None,
    colour: str = 'lightseagreen',
    colour_compare: str = 'gold'
):
    """Plot a profile as a lollipop plot.

    If two profiles are provided, the plot compares the two and adds a legend
    at the bottom of the plot.

    Parameters
    ----------
    profile : Dict[str, float]
        Primary profile to display.
    profile_compare : Dict[str, float], optional
        Secondary profile. If it is not empty, the primary profile is compared
        to the secondary profile, by default None.
    profile_name : str, optional
        Name of the primary profile, by default None.
    profile_compare_name : str, optional
        Name of the secondary profile, by default None.
    colour : str, optional
       Colour of the lollipop for the primary profile,
       by default 'lightseagreen'.
    colour_compare : str, optional
        Colour of the lollipop for the secondary profile, by default 'gold'.
    """
    right_side_labels = []
    left_side_labels = []
    x1 = []
    x2 = []

    # prepare data
    for i in range(NUM_AXIS):
        p = list(IDEAL_PROFILES.keys())[i]
        p_inv = get_opposite_profile(p)
        left_side_labels.append(p_inv)
        right_side_labels.append(p)
        try:
            value = (profile[p]-profile[p_inv]) / (profile[p]+profile[p_inv])
        except ZeroDivisionError:
            value = 0.
        x1.append(value)
        if profile_compare:
            try:
                value = (profile_compare[p]-profile_compare[p_inv]) / \
                    (profile_compare[p]+profile_compare[p_inv])
            except ZeroDivisionError:
                value = 0.
            x2.append(value)
    right_side_labels.reverse()
    left_side_labels.reverse()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_box_aspect(0.55)
    y = range(NUM_AXIS-1, -1, -1)

    # make lollipop with horizontal lines and scatter plots
    if profile_compare:
        y1 = [yi + 0.1 for yi in y]
        ax.scatter(x1, y1, s=550, c=colour, zorder=1, label=profile_name)
        ax.hlines(y1, [0]*NUM_AXIS, x1, colors='darkgray', lw=5, zorder=0)
        y2 = [yi - 0.1 for yi in y]
        ax.scatter(x2, y2, s=550, c=colour_compare, zorder=1,
                   label=profile_compare_name)
        ax.hlines(y2, [0]*NUM_AXIS, x2, colors='darkgray', lw=5, zorder=0)
        ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.16))
    else:
        ax.scatter(x1, y, s=550, c=colour, zorder=1)
        ax.hlines(y, [0]*NUM_AXIS, x1, colors='darkgray', lw=5, zorder=0)

    # render axis
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.25, 3.25)
    ax.vlines(0, -0.25, 3.25, zorder=-1, colors='lightgray', lw=2)
    ax.vlines(
        [0.2*i for i in range(-6, 6)], -0.25, 3.25,
        linestyle='--', color='lightgray', zorder=-3
    )
    ax.set_xticks([])
    ax.set_yticks(range(NUM_AXIS))
    ax.set_yticklabels(left_side_labels)

    # make secondary y-axis for complementary profile labels
    ax_right = ax.twinx()
    ax.set_box_aspect(0.55)
    ax_right.set_ylim(-0.25, 3.25)
    ax_right.set_yticks(range(NUM_AXIS))
    ax_right.set_yticklabels(right_side_labels)

    return fig


if __name__ == '__main__':

    print("hmmm")
    # Basic usage for integration with the server
    pdata =   {
        "Q1_group": "outgroup,ingroup",
        "Q2_group": "ingroup,ingroup",
        "Q3_group": "outgroup,outgroup",
        "Q4_group": "outgroup,ingroup",
        "Q5_group": "outgroup,outgroup",
        "Q6_group": "outgroup,outgroup",
        "Q7_group": "ingroup,ingroup",
        "Q8_group": "outgroup,outgroup",
        "Q9_group": "ingroup,outgroup",
        "Q10_group": "ingroup,outgroup",
        "Q11_group": "ingroup,ingroup",
        "Q12_group": "ingroup,ingroup",
        "id": "5c9d92f871ad820001da9510",
        "Q1_values": "13,13",
        "Q2_values": "16,19",
        "Q3_values": "18,15",
        "Q4_values": "13,13",
        "Q5_values": "14,15",
        "Q6_values": "12,15",
        "Q7_values": "12,11",
        "Q8_values": "21,25",
        "Q9_values": "18,19",
        "Q10_values": "16,19",
        "Q11_values": "12,15",
        "Q12_values": "20,23"
    }

    res = analyze_participant(pdata, 'equidistant_ranks')
    print(res)


    # Analyse made-up data
    games = [
        Game.from_csv(f"example_games/matrix{i}.csv", delimiter=';', \
                      game_id=f"Game {i}")
        for i in range(1, 5)
    ]

    _ = [g.assign_groups('in', 'out') for g in games]

    # compute profiles for a "large" number of players in parallel
    # there is not a big number of participants here, but this is a template for
    # future analysis
    fake_participants = {
        'egalitarian':                          [6, 6, 6, 6],
        'differentatian-altruist-utilitarian':  [12, 12, 12, 12],
        'badperson-sociopath':                  [0, 0, 0, 0],
        'sectarian':                            [0, 0, 12, 12],
        'self-hater-differentarian':            [12, 12, 0, 0]
    }

    num_cores = mp.cpu_count()
    num_participants = len(fake_participants)
    if num_participants < num_cores:
        chunksize = 1
        workers = num_participants
    else:
        chunksize = num_participants // num_cores
        workers = num_cores

    for m in METHODS:
        args = [
            (games, choices, m, f'plots/{name}_{m}.png')
            for name, choices in fake_participants.items()
        ]
        pool = mp.Pool(workers)
        it = pool.starmap(compute_plot_participant_profile, args, chunksize)
        pool.close()

    # Example: Compare randomly generated profiles in a plot
    import random
    random.seed(10)

    print("what about this")
    random_profile1 = {
        label: random.uniform(0, 1) for label in IDEAL_PROFILES.keys()
    }
    random_profile2 = {
        label: random.uniform(0, 1) for label in IDEAL_PROFILES.keys()
    }
    fig = lollipop_plot(
        random_profile1,
        random_profile2,
        profile_name='Random 1', 
        profile_compare_name='Random 2'
    )

    fig.savefig("plots/comparison_example2.png", dpi=400)
