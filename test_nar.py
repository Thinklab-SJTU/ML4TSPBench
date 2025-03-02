from models.modelzoo import TSPNARTrainer, get_nar_model
from utils.args import nar_test_arg_parser


def test_nar(test_nar_args):
    model_class = get_nar_model(test_nar_args.task, test_nar_args.encoder)
    model = model_class(**vars(test_nar_args))
    trainer = TSPNARTrainer(model=model, **vars(test_nar_args))
    trainer.model_test()


if __name__ == '__main__':
    test_nar_args = nar_test_arg_parser()
    test_nar(test_nar_args)