from typing import Optional

from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class L1ProjectedGradientDescentAttack(L1BaseGradientDescent):
    """L1 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        att_def_avg: Optional[str] = None,
        follow_dir: Optional[bool] = False,
        max_val: Optional[bool] = None,
        rand_div: Optional[bool] = None,
        variant: Optional[str] = 'v2',
        max_opp_dir: Optional[bool] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg=att_def_avg,
            follow_dir=follow_dir,
            max_val=max_val,
            rand_div=rand_div,
            variant=variant,
            max_opp_dir=max_opp_dir,
            weight1=weight1,
            weight2=weight2,
        )


class L2ProjectedGradientDescentAttack(L2BaseGradientDescent):
    """L2 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        att_def_avg: Optional[str] = None,
        follow_dir: Optional[bool] = False,
        max_val: Optional[bool] = None,
        rand_div: Optional[bool] = None,
        variant: Optional[str] = 'v2',
        max_opp_dir: Optional[bool] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg=att_def_avg,
            follow_dir=follow_dir,
            max_val=max_val,
            rand_div=rand_div,
            variant=variant,
            max_opp_dir=max_opp_dir,
            weight1=weight1,
            weight2=weight2,
        )


class LinfProjectedGradientDescentAttack(LinfBaseGradientDescent):
    """Linf Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.01 / 0.3,
        abs_stepsize: Optional[float] = None,
        steps: int = 40,
        random_start: bool = True,
        att_def_avg: Optional[str] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
        follow_dir: Optional[bool] = False,
        max_val: Optional[bool] = None,
        rand_div: Optional[bool] = None,
        variant: Optional[str] = 'v2',
        max_opp_dir: Optional[bool] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg=att_def_avg,
            follow_dir=follow_dir,
            max_val=max_val,
            rand_div=rand_div,
            variant=variant,
            max_opp_dir=max_opp_dir,
            weight1=weight1,
            weight2=weight2,
        )
