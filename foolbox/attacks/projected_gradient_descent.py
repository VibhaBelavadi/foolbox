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
        att_def_avg_third: Optional[str] = None,
        att_def_avg_fourth: Optional[str] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg_third=att_def_avg_third,
            att_def_avg_fourth=att_def_avg_fourth,
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
        att_def_avg_third: Optional[str] = None,
        att_def_avg_fourth: Optional[str] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg_third=att_def_avg_third,
            att_def_avg_fourth=att_def_avg_fourth,
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
        att_def_avg_third: Optional[str] = None,
        att_def_avg_fourth: Optional[str] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            att_def_avg_third=att_def_avg_third,
            att_def_avg_fourth=att_def_avg_fourth,
            weight1=weight1,
            weight2=weight2,
        )
