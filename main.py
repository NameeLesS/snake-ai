import argparse
import torch
import os


from game import Game, GameEnviroment
from distributed import DataCollectionProcess
from network import DQN
from config import *


def load_model(model_path):
    predict_model = DQN()
    predict_model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=torch.device('cpu')))
    return predict_model


def do_one_step(game, model):
    state = DataCollectionProcess.preprocess_state(game.get_state()).to(torch.float32)

    model.eval()
    with torch.no_grad():
        action = torch.argmax(model(state))

    game.step(int(action))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='store_true')
    args = parser.parse_args()
    if args.model:
        game = GameEnviroment(size=SCREEN_SIZE, fps=FPS, training_mode=False)
        game.execute()
        model = load_model('/home/nameless/Downloads/')
        while True:
            do_one_step(game, model)

    else:
        game = Game(size=SCREEN_SIZE, fps=FPS)
        game.execute()


if __name__ == '__main__':
    main()
