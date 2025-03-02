from models.modelzoo import TSPNARTrainer, TSPNAREncoder, get_nar_model
from utils.args import nar_train_arg_parser


def train_nar(train_nar_args):
    model_class = get_nar_model(train_nar_args.task, train_nar_args.encoder)
    model = model_class(**vars(train_nar_args))
    model: TSPNAREncoder
    if train_nar_args.ckpt_path is not None:
        model.load_ckpt(train_nar_args.ckpt_path)
    trainer = TSPNARTrainer(model=model, **vars(train_nar_args))
    trainer.model_train()


if __name__ == '__main__':
    train_nar_args = nar_train_arg_parser()
    train_nar(train_nar_args)