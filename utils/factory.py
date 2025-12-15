def get_model(model_name, args):
    name = model_name.lower()
    if name == 'mote':
        from models.mote import Learner
    elif name == 'mote_limit':
        from models.mote_limit import Learner
    elif name == 'mote_seg':
        from models.mote_seg import SegLearner as Learner
    else:
        assert 0, f"Unknown model name: {model_name}"
    return Learner(args)