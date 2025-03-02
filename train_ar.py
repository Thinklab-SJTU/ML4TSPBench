from models.modelzoo import TSPARTrainer, TSPAREncoder, get_ar_model
from utils.args import ar_train_arg_parser


def train_ar(train_ar_args):
    model_class = get_ar_model(train_ar_args.task, train_ar_args.encoder)
    model = model_class(**vars(train_ar_args))
    model: TSPAREncoder
    if train_ar_args.ckpt_path is not None:
        model.load_ckpt(train_ar_args.ckpt_path)
    trainer = TSPARTrainer(model=model, **vars(train_ar_args))
    trainer.model_train()


if __name__ == '__main__':
    train_ar_args = ar_train_arg_parser()
    train_ar(train_ar_args)