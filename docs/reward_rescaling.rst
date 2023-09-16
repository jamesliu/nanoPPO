Reward Rescaling
================

Reward scaling is an essential topic in Reinforcement Learning (RL), and it can have a profound impact on the training dynamics.

Purpose of Reward Rescaling
---------------------------

1. **Stability**: 
    Normalizing rewards can lead to more stable training, especially in algorithms that use gradient-based optimization like PPO. Large rewards can lead to large updates, which can be harmful to training.
  
2. **Comparability**: 
    By rescaling, it's easier to compare results across different environments or scenarios. If you're tweaking an environment or comparing multiple environments, having a consistent reward scale can be helpful.
  
3. **Convergence Speed**: 
    Algorithms often converge faster when rewards are normalized because the optimization landscape becomes more consistent.

Dealing with Sparse and Skewed Rewards
--------------------------------------

In environments like `MountainCar`, the reward distribution is indeed sparse and skewed. This poses challenges:

1. **Sparse Rewards**: 
    Most of the rewards are negative and small, but occasionally there's a large positive reward.
    
2. **Skewed Distribution**: 
    When you normalize such rewards using mean and standard deviation, the large reward will not be as influential as you might want because it's an outlier.

Strategies to Handle Sparse and Skewed Rewards
----------------------------------------------

1. **Clipping**: 
    Instead of just normalizing, you can also clip the rewards between two values (e.g., -1 and 1). This ensures that no reward becomes too dominant. The downside is that you lose information about the magnitude of the reward.

2. **Moving Average Rescaling**: 
    Instead of computing the mean and standard deviation over all rewards ever seen, you can compute them over a moving window. This can make the normalization more sensitive to recent rewards, which might be desirable in non-stationary environments.

3. **No Rescaling**: 
    In some cases, especially when the reward structure is essential, and you don't want to lose the reward magnitude information, you might choose not to rescale at all. However, you should ensure that your learning rates and other hyperparameters are set appropriately to handle the reward magnitudes.

4. **Custom Rescaling**: 
    You could use domain knowledge to design a custom rescaling function. For example, you could rescale all negative rewards to be between -1 and 0 and all positive rewards between 0 and 1, ensuring that the positive reward's relative magnitude is preserved.

Should you rescale in `MountainCar`?
------------------------------------

Given the nature of `MountainCar`, where most rewards are small and negative, but the successful completion of the task results in a large positive reward, you have a few options:

1. **No Rescaling**: 
    You could opt not to rescale and adjust other hyperparameters to handle the magnitude of rewards.
    
2. **Clipping**: 
    After normalizing, clip the rewards between [-1, 1]. This way, the positive reward will have a maximum influence on the update, which might be desirable.
    
3. **Custom Rescaling**: 
    If you feel the large positive reward is not getting enough emphasis in the training, use domain knowledge to come up with a custom scaling that amplifies the positive reward.

Conclusion
----------

The best strategy often depends on the specific problem, the RL algorithm used, and the desired behavior. In cases with skewed rewards, it's essential to understand the implications of reward scaling and make informed decisions. Experimentation is crucial. You might want to try different strategies and see which one leads to better and more stable training.
