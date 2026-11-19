"""Validation checkpoint selection, independent from torch and GPU execution."""

from dataclasses import dataclass, asdict


@dataclass
class Selection:
    baseline_clean: float
    clean_tolerance_pp: float = 1.0
    min_delta_pp: float = 0.1
    patience: int = 4
    min_epochs: float = 2.0
    best_lr4: float = -1.0
    progress_lr4: float = -1.0
    best_examples: int = 0
    bad_checks: int = 0

    def observe(self, lr4, clean, examples, epoch):
        import math
        if not all(math.isfinite(x) for x in (lr4,clean,epoch)):
            raise ValueError('Nonfinite validation metric')
        eligible=clean >= self.baseline_clean-self.clean_tolerance_pp
        save_best=eligible and lr4>self.best_lr4
        if save_best:self.best_lr4,self.best_examples=lr4,examples
        if eligible and lr4>=self.progress_lr4+self.min_delta_pp:
            self.progress_lr4=lr4;self.bad_checks=0
        else:self.bad_checks+=1
        return {'eligible':eligible,'save_best':save_best,
                'stop':epoch>=self.min_epochs and self.bad_checks>=self.patience}

    def state_dict(self):return asdict(self)
