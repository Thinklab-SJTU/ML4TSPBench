from models.modelzoo import TSPARTrainer, get_ar_model
from utils.args import ar_test_arg_parser


def test_ar(test_ar_args):
    model_class = get_ar_model(test_ar_args.task, test_ar_args.encoder)
    model = model_class(**vars(test_ar_args))
    trainer = TSPARTrainer(model=model, **vars(test_ar_args))
    trainer.model_test()


if __name__ == '__main__':
    test_ar_args = ar_test_arg_parser()
    test_ar(test_ar_args)