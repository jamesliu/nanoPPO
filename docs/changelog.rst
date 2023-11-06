.. _changelog:

===========
 Changelog
===========

.. _v0_15:

0.15 (2023-11-06)
-----------------

- Adding Actor Critic Causal Attention Policy
- Adding Cosine Annealing LR Scheduler
- Adding gradient and weight NAN and INF check
- Adding a debug flag when detecting NAN and INF 

.. _v0_14:

0.14 (2023-10-15)
-----------------

- Adding action rescaling: rescale actions
- Adding train_reward to metrics dictionary
- Adding version number in __init__.py
- Adding device configuration: facilitating both CPU and GPU configuration.
- Adding gradient clipping: introduced gradient clipping with clip grad norm to prevent updates that are too aggressive.

.. _v0_13:

0.13 (2023-09-19)
-----------------

- Creating a ppo agent training on continuous action space
- Adding two environments: `Pendulum-v1` and `MountainCarContinuous-v0`
- Creating two new environments: `PointMass1D-v0` and `PointMass2D-v0`
- Creating test cases: test_gae.py, test_returns_and_advantages.py, test_normalizer.py, test_random_utils.py , test_reward_scaler.py

.. _v0_11:

0.11 (2023-07-03)
-----------------

- Creating a ppo agent training on discrete action space
- Adding one environment: `CartPole-v1`

.. _v0_1:

0.1 (2023-07-01)
----------------

- Initial release to PyPI
