from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import Bounds

from ..models.base import Model

from ..criteria import Misclassification, TargetedMisclassification

from ..distances import l1, l2, linf

from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs


class BaseGradientDescent(FixedEpsilonAttack, ABC):
    def __init__(
        self,
        *,
        rel_stepsize: float,
        abs_stepsize: Optional[float] = None,
        steps: int,
        random_start: bool,
        follow_dir: Optional[bool] = False,
        max_val: Optional[bool] = None,
        rand_div: Optional[bool] = None,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.follow_dir = follow_dir
        self.max_val = max_val
        self.rand_div = rand_div

    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()

        return loss_fn

    def value_and_grad(
        # can be overridden by users
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(
        self,
        model1: Model,
        model2: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        classes1, classes2 = None, None
        del inputs, criterion, kwargs

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes1 = criterion_.labels1
            classes2 = criterion_.labels2
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn1 = self.get_loss_fn(model1, classes1)
        loss_fn2 = self.get_loss_fn(model2, classes2)

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model1.bounds)
        else:
            x = x0

        # Tried a variant where the gradients are plainly added after normalization: explained why it was wrong
        # What if I: 1. normalize g1 and g2 2. add/max/min 3. project alpha*_ into lp norm 4. add to x 5. bound x
        for _ in range(self.steps):
            _, gradients1 = self.value_and_grad(loss_fn1, x)
            _, gradients2 = self.value_and_grad(loss_fn2, x)

            # label flip: when only adding the two gradients
            # gradients1 = self.normalize(gradients1, x=x, bounds=model1.bounds)
            # gradients2 = self.normalize(gradients2, x=x, bounds=model2.bounds)
            # x = x + gradient_step_sign * stepsize * (gradients1 + gradients2)
            # This approach is wrong since (gradients1 + gradients2)
            # is not bound by lp norm

            # this approach is wrong since gradients_max*g_same_dir
            # is not bound by lp norm
            # x = x + gradient_step_sign * stepsize * g_same_dir * gradients_max

            # check if we need to follow along direction
            if self.follow_dir:
                g_same_dir = gradients1.sign() + gradients2.sign()
                g_opp_dir = gradients1.sign() - gradients2.sign()
            else:
                g_same_dir, g_opp_dir = 1, 1

            # version1: adding of two gradients
            # version 2 and 3: get the same/opposite direction elements of two tensors
            if self.max_val is None:
                final_gradients = gradients1 + gradients2
            elif self.max_val is False:
                final_gradients = ep.minimum(gradients1, gradients2)
            else:
                final_gradients = ep.maximum(gradients1, gradients2)

            # intializing with random noise in opposite dir
            # There is another variant that needs to be tried
            if self.rand_div is None:
                final_gradients = final_gradients * g_same_dir
            else:
                rand_init = self.get_random_start(x0, epsilon)
                rand_init = ep.clip(rand_init, *model1.bounds)/self.rand_div
                """
                # Try this
                # final_gradients = final_gradients * g_same_dir
                # final_gradients = self.normalize(final_gradients, x=x, bounds=model1.bounds)
                # x = x + gradient_step_sign * stepsize * final_gradients + rand_init * g_opp_dir
                """
                final_gradients = final_gradients * g_same_dir + rand_init * g_opp_dir

            # normalize the updated gradient before feeding it to the model
            final_gradients = self.normalize(final_gradients, x=x, bounds=model1.bounds)
            x = x + gradient_step_sign * stepsize * final_gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model1.bounds)

            del final_gradients, gradients2, gradients1, g_same_dir, g_opp_dir

        return restore_type(x)

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

    @abstractmethod
    def normalize(
            self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        ...

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...


def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    # https://mathoverflow.net/a/9188
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, : n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x


def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s


def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    """Sampling from the n-ball

    Implementation of the algorithm proposed by Voelker et al. [#Voel17]_

    References:
        .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
            from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    s = uniform_l2_n_spheres(dummy, batch_size, n + 1)
    b = s[:, :n]
    return b


class L1BaseGradientDescent(BaseGradientDescent):
    distance = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
            self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)


class L2BaseGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
            self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)


class LinfBaseGradientDescent(BaseGradientDescent):
    distance = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(
            self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return gradients.sign()

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
