from gym.envs.registration import register

register(
    id='recaptcha-v0',
    entry_point='recaptcha.envs:Recaptcha',
)