import math

class CosineLRScheduler:
    def __init__(self, warmup_iters, lr_decay_iters, learning_rate, min_lr):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.optimizer = None  # Placeholder for the optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_lr_actor(self, it):
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def get_lr_critic(self, it):
        # You can define a separate LR schedule for the critic if needed
        # Example: return self.learning_rate_critic * it / self.warmup_iters
        return self.get_lr_actor(it)

    def step(self):
        if self.optimizer is None:
            raise ValueError("Optimizer has not been set.")
        
        for param_group in self.optimizer.param_groups:
            actor_lr = self.get_lr_actor(param_group['iterations'])
            param_group['lr'] = actor_lr
