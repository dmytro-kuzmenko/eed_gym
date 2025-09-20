def test_import_and_step():
    from eed_benchmark.envs.empathic_disobedience_env import EmpathicDisobedienceEnv, COMPLY
    env = EmpathicDisobedienceEnv(seed=0)
    obs, info = env.reset()
    assert len(obs) > 0
    obs, rew, term, trunc, info = env.step(COMPLY)
    assert isinstance(rew, float)
