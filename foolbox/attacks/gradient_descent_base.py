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
        att_def_avg_third: Optional[str] = None,
        att_def_avg_fourth: Optional[str] = None,
        weight1: Optional[float] = None,
        weight2: Optional[float] = None,
        follow_dir: Optional[bool] = False,
        max_val: Optional[bool] = None,
        rand_div: Optional[bool] = None,
        max_opp_dir: Optional[bool] = None,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.att_def_avg_third = att_def_avg_third
        self.att_def_avg_fourth = att_def_avg_fourth
        self.follow_dir = follow_dir
        self.max_val = max_val
        self.rand_div = rand_div
        self.max_opp_dir = max_opp_dir
        self.weight1 = weight1
        self.weight2 = weight2

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
        model3: Optional[Model] = None,
        model4: Optional[Model] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        classes1, classes2 = None, None
        classes3, classes4, loss_fn3, loss_fn4 = None, None, None, None
        del inputs, criterion, kwargs

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        # our criterion is Misclassification
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes1 = criterion_.labels1
            classes2 = criterion_.labels2
            classes3 = criterion_.labels3
            classes4 = criterion_.labels4
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn1 = self.get_loss_fn(model1, classes1)
        loss_fn2 = self.get_loss_fn(model2, classes2)

        if classes3 is not None:
            loss_fn3 = self.get_loss_fn(model3, classes3)

        if classes4 is not None:
            loss_fn4 = self.get_loss_fn(model4, classes4)

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model1.bounds)
        else:
            x = x0

        if self.weight1 is None:
            weight1 = 1
        else:
            weight1 = self.weight1

        if self.weight2 is None:
            weight2 = 1
        else:
            weight2 = self.weight2

        x = self.sum_label_flip(x, x0, epsilon, stepsize, loss_fn1, model1, loss_fn2, model2,
                                loss_fn3, model3, loss_fn4, model4, gradient_step_sign, weight1, weight2)

        # TODO: Remove this obselete piece of code
        """
        if self.max_val is None:
            if self.max_opp_dir is None:
                x = self.sum_label_flip(x, x0, epsilon, stepsize, loss_fn1, model1, loss_fn2, model2,
                                loss_fn3, model3, loss_fn4, model4, gradient_step_sign, weight1, weight2)
            else:
                x = self.sum_label_flip_min_max(x, x0, epsilon, stepsize, loss_fn1, model1, loss_fn2, model2,
                                                gradient_step_sign, weight1, weight2)
        else:
            x = self.minmax_label_flip(x, x0, epsilon, stepsize, loss_fn1, model1, loss_fn2, model2, gradient_step_sign)
        """
        return restore_type(x)

    def sum_label_flip(self, x, x0, epsilon, stepsize, loss_fn1, model_1, loss_fn2, model_2, loss_fn3=None,
                       model_3=None, loss_fn4=None, model_4=None, gradient_step_sign=1.0, weight_1=1.0, weight_2=1.0):

        for num_steps in range(self.steps):
            # TODO: remove this obselete code
            # same_dir, opp_dir = 1, 1

            _, gradients_1 = self.value_and_grad(loss_fn1, x)
            _, gradients_2 = self.value_and_grad(loss_fn2, x)
            gradients_1 = self.normalize(gradients_1, x=x, bounds=model_1.bounds)
            gradients_2 = self.normalize(gradients_2, x=x, bounds=model_2.bounds)

            if loss_fn3 is not None and loss_fn4 is not None:
                _, gradients_3 = self.value_and_grad(loss_fn3, x)
                gradients_3 = self.normalize(gradients_3, x=x, bounds=model_3.bounds)

                _, gradients_4 = self.value_and_grad(loss_fn4, x)
                gradients_4 = self.normalize(gradients_4, x=x, bounds=model_4.bounds)

                attack_gradients, defend_gradients = None, None
                if self.att_def_avg_third == 'att' and self.att_def_avg_fourth == 'att':
                    attack_gradients = (gradients_1 + gradients_3 + gradients_4)/3
                    defend_gradients = gradients_2
                    final_gradients = weight_1 * attack_gradients + weight_2 * defend_gradients

                elif self.att_def_avg_third == 'att' and self.att_def_avg_fourth == 'def':
                    attack_gradients = (gradients_1 + gradients_3)/2
                    defend_gradients = (gradients_2 + gradients_4)/2
                    final_gradients = weight_1 * attack_gradients + weight_2 * defend_gradients

                elif self.att_def_avg_third == 'def' and self.att_def_avg_fourth == 'att':
                    attack_gradients = (gradients_1 + gradients_4)/2
                    defend_gradients = (gradients_2 + gradients_3)/2
                    final_gradients = weight_1 * attack_gradients + weight_2 * defend_gradients

                elif self.att_def_avg_third == 'def' and self.att_def_avg_fourth == 'def':
                    attack_gradients = gradients_1
                    defend_gradients = (gradients_2 + gradients_3 + gradients_4) / 3
                    final_gradients = weight_1 * attack_gradients + weight_2 * defend_gradients

                else:
                    final_gradients = gradients_4+gradients_3+gradients_2+gradients_1
                del gradients_3, gradients_4
            elif loss_fn3 is not None and loss_fn4 is None:
                _, gradients_3 = self.value_and_grad(loss_fn3, x)
                gradients_3 = self.normalize(gradients_3, x=x, bounds=model_3.bounds)

                if self.att_def_avg_third == 'att':
                    final_gradients = weight_1*(gradients_3+gradients_1)/2 + weight_2*gradients_2

                elif self.att_def_avg_third == 'def':
                    final_gradients = weight_2*(gradients_3+gradients_2)/2 + weight_1*gradients_1

                else:
                    final_gradients = gradients_3+gradients_2+gradients_1

                del gradients_3
            elif loss_fn4 is not None and loss_fn3 is None:
                _, gradients_4 = self.value_and_grad(loss_fn4, x)
                gradients_4 = self.normalize(gradients_4, x=x, bounds=model_4.bounds)

                if self.att_def_avg_fourth == 'att':
                    final_gradients = weight_1*(gradients_4 + gradients_1)/2 + weight_2*gradients_2

                elif self.att_def_avg_fourth == 'def':
                    final_gradients = weight_2*(gradients_4 + gradients_2)/2 + weight_1*gradients_1

                else:
                    final_gradients = gradients_4+gradients_2+gradients_1

                del gradients_4
            else:
                final_gradients = weight_1*gradients_1 + weight_2*gradients_2

            # TODO: remove this obselete code reference
            """
            # find elements of gradients_1 and gradients_2 in the same direction
            if self.follow_dir:
                same_dir = gradients_1.sign() + gradients_2.sign()
                same_dir = ep.abs(same_dir)/2  # get the absolute value of this mask
                opp_dir = gradients_1.sign() - gradients_2.sign()
                opp_dir = ep.abs(opp_dir)/2  # get the absolute value of this mask

            final_gradients = final_gradients * same_dir

            # normalize random initialization in the opposite direction
            if self.rand_div is None:
                rand_init = 0
            else:
                rand_init = self.get_random_start(x0, epsilon)
                rand_init = ep.clip(rand_init, *model_1.bounds) / self.rand_div

            x = x + gradient_step_sign * stepsize * final_gradients + rand_init * opp_dir
            """

            x = x + gradient_step_sign * stepsize * final_gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model_1.bounds)

            del final_gradients, gradients_2, gradients_1, same_dir, opp_dir, rand_init

        return x

    # Note: OBSELETE code....no longer used
    # TODO: Remove references to all the variables in this obselete code
    """
    def sum_label_flip_min_max(self, x, x0, epsilon, stepsize, loss_fn1, model_1, loss_fn2, model_2,
                               gradient_step_sign=1.0, weight_1=1.0, weight_2=1.0):

        for num_steps in range(self.steps):
            _, gradients_1 = self.value_and_grad(loss_fn1, x)
            _, gradients_2 = self.value_and_grad(loss_fn2, x)

            gradients_1 = self.normalize(gradients_1, x=x, bounds=model_1.bounds)
            gradients_2 = self.normalize(gradients_2, x=x, bounds=model_2.bounds)

            final_gradients = weight_1*gradients_1 + weight_2*gradients_2

            same_dir = gradients_1.sign() + gradients_2.sign()
            same_dir = ep.abs(same_dir)/2  # get the absolute value of this mask
            opp_dir = gradients_1.sign() - gradients_2.sign()
            opp_dir = ep.abs(opp_dir)/2  # get the absolute value of this mask

            if self.max_opp_dir is True:
                max_grad = ep.maximum(gradients_1, gradients_2)
                x = x + gradient_step_sign * stepsize * (final_gradients*same_dir + max_grad*opp_dir)
                del max_grad
            else:
                min_grad = ep.minimum(gradients_1, gradients_2)
                x = x + gradient_step_sign * stepsize * (final_gradients*same_dir + min_grad*opp_dir)
                del min_grad

            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model_1.bounds)

            del final_gradients, gradients_2, gradients_1, same_dir, opp_dir

        return x

    def minmax_label_flip(self, x, x0, epsilon, stepsize, loss_fn1, model_1, loss_fn2, model_2,
                          gradient_step_sign=1.0):

        for num_steps in range(self.steps):
            same_dir, opp_dir = 1, 1
            _, gradients_1 = self.value_and_grad(loss_fn1, x)
            _, gradients_2 = self.value_and_grad(loss_fn2, x)

            gradients_1 = self.normalize(gradients_1, x=x, bounds=model_1.bounds)
            gradients_2 = self.normalize(gradients_2, x=x, bounds=model_2.bounds)

            if self.max_val is False:
                final_gradients = ep.minimum(gradients_1, gradients_2)
            else:
                final_gradients = ep.maximum(gradients_1, gradients_2)

            if self.follow_dir:
                same_dir = gradients_1.sign() + gradients_2.sign()
                same_dir = ep.abs(same_dir)/2  # get the absolute value of this mask
                opp_dir = gradients_1.sign() - gradients_2.sign()
                opp_dir = ep.abs(opp_dir)/2  # get the absolute value of this mask

            if self.rand_div is None:
                rand_init = 0
            else:
                rand_init = self.get_random_start(x0, epsilon)
                rand_init = ep.clip(rand_init, *model_1.bounds) / self.rand_div

            x = x + gradient_step_sign * stepsize * final_gradients * same_dir + rand_init * opp_dir
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model_1.bounds)

            del final_gradients, gradients_2, gradients_1, same_dir, opp_dir, rand_init

        return x
    """

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
