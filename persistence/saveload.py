import d3rlpy
from d3rlpy.models.encoders import register_encoder_factory


def load_with_custom_encoder(CustomEncoderFactoryClass, modelpath: str):
    # register your own encoder factory
    register_encoder_factory(CustomEncoderFactoryClass)

    # load algorithm from d3
    algo_model = d3rlpy.load_learnable(modelpath)
    return algo_model