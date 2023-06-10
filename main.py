import argparse
import torch
import os

import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T

from game import Game, GameEnviroment
from network import DQN
from config import *


def load_model(model_path):
    predict_model = DQN()
    predict_model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=torch.device('cpu')))
    return predict_model


def do_one_step(game, model):
    state = F.to_pil_image(game.get_state())
    state = F.to_grayscale(state)
    state = F.resize(state, INPUT_SIZE[1:3], interpolation=T.InterpolationMode.NEAREST_EXACT)
    state = F.pil_to_tensor(state)
    state = state.squeeze().reshape((1, *INPUT_SIZE)).to(torch.float32)

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
