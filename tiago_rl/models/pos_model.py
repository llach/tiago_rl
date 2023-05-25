from tiago_rl import safe_rescale

class PosModel:

    def predict(self, obs, **kwargs):
        q = safe_rescale(obs[:2], [-1, 1], [0.0, 0.045])
        qdelta = safe_rescale(obs[2:4], [-1, 1], [-0.045, 0.045])
        qdes = q+qdelta
        
        return safe_rescale(qdes, [0.0, 0.045], [-1, 1]), {}