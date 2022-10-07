# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import socket

import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List

import asr_pb2

import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import defaultdict
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder

from gevent import monkey

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import (
    MicroRTSGridModeSharedMemVecEnv as MicroRTSGridModeVecEnv,
)

# monkey.patch_all()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=5000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--n-minibatch', type=int, default=4,
        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=0,
        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--num-models', type=int, default=100,
        help='the number of models saved')
    parser.add_argument('--max-eval-workers', type=int, default=4,
        help='the maximum number of eval workers (skips evaluation when set to 0)')
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during training')
    parser.add_argument('--eval-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during evaluation')

    arguments = parser.parse_args()
    if not arguments.seed:
        arguments.seed = int(time.time())
    arguments.num_envs = arguments.num_selfplay_envs + arguments.num_bot_envs
    arguments.batch_size = int(arguments.num_envs * arguments.num_steps)
    arguments.minibatch_size = int(arguments.batch_size // arguments.n_minibatch)
    arguments.num_updates = arguments.total_timesteps // arguments.batch_size
    arguments.save_frequency = max(1, int(arguments.num_updates // arguments.num_models))
    # fmt: on
    return arguments


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i, done in enumerate(dones):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if done:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, mask_value=None):
        if masks is None:
            masks = []
        logits = torch.where(masks.bool(), logits, mask_value)
        super().__init__(probs, logits, validate_args)

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super().__init__()
        self.mapsize = mapsize
        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action_and_value(self, x, action=None, invalid_action_masks=None, envs=None, num_envs=24, device=None, sock=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)

        if action is None:
            # invalid_action_masks [24, 256, 78]
            # presplit_invalid_action_masks ([24, 256, 6], [24, 256, 4], [24, 256, 4], [24, 256, 4], [24, 256, 4], [24, 256, 7], [24, 256, 49])
            #time_start = time.perf_counter_ns()
            presplit_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=2)
            #time_stop = time.perf_counter_ns()
            #print("Runtime for presplit_invalid_action_masks: ", (time_stop - time_start)/1000000, "ms")

            # for every of the 24 games, send the game state one by one to the constraint solver
            for game in range(num_envs):
                #time_start = time.perf_counter_ns()
                game_state = asr_pb2.State()
                filtered_actions = torch.zeros(256, 78).to(device)
                number_units = 0
                total_number_possible_actions = 0;
                for cell in {c for c in torch.where(presplit_invalid_action_masks[0][game]==1.0)[0].to('cpu').numpy()}:
                    #time_start_cell = time.perf_counter_ns()
                    action_type = presplit_invalid_action_masks[0][game][cell] # NOOP, move, harvest, return, produce, attack
                    move_direction = presplit_invalid_action_masks[1][game][cell] # north, east, south, west
                    harvest_direction = presplit_invalid_action_masks[2][game][cell] # north, east, south, west
                    return_direction = presplit_invalid_action_masks[3][game][cell] # north, east, south, west
                    produce_direction = presplit_invalid_action_masks[4][game][cell] # north, east, south, west
                    produce_type = presplit_invalid_action_masks[5][game][cell] # resource, base, barrack, worker, light, heavy, ranged
                    attack_position = presplit_invalid_action_masks[6][game][cell] # [0,48]
                    #time_stop_cell = time.perf_counter_ns()
                    #print("Runtime to read cell informations: ", (time_stop_cell - time_start_cell)/1000000, "ms")

                    #time_start_if_action = time.perf_counter_ns()
                    unit = game_state.units.add()
                    number_units = number_units + 1
                    unit.unit_id = cell
                    if action_type[0] == 1.0: # NOOP
                        unit.actions_id.extend([1]) # value 1
                        total_number_possible_actions = total_number_possible_actions + 1
                    if action_type[1] == 1.0: # Move
                        unit.actions_id.extend([1+i+1 for i in torch.where(move_direction == 1.0)[0]]) # value in [2,5] 
                        total_number_possible_actions = total_number_possible_actions + 1
                    if action_type[2] == 1.0: # Harvest
                        unit.actions_id.extend([5+i+1 for i in torch.where(harvest_direction == 1.0)[0]]) # value in [6,9]
                        total_number_possible_actions = total_number_possible_actions + 1
                    if action_type[3] == 1.0: # Return to base with minerals
                        unit.actions_id.extend([9+i+1 for i in torch.where(return_direction == 1.0)[0]]) # value in [10,13]
                        total_number_possible_actions = total_number_possible_actions + 1
                    if action_type[4] == 1.0 and 1.0 in produce_direction and 1.0 in produce_type: # Produce
                        for d in torch.where(produce_direction == 1.0)[0]:
                            unit.actions_id.extend([13 + 7*d + t + 1 for t in torch.where(produce_type == 1.0)[0]]) # value in [14,41]
                            total_number_possible_actions = total_number_possible_actions + 1
                    if action_type[5] == 1.0: # Attack
                        unit.actions_id.extend([41+i+1 for i in torch.where(attack_position == 1.0)[0]]) # value in [42,90]
                        total_number_possible_actions = total_number_possible_actions + 1
                    #time_stop_if_action = time.perf_counter_ns()
                    #print("Runtime to convert actions: ", (time_stop_if_action - time_start_if_action)/1000000, "ms")

                #time_stop = time.perf_counter_ns()
                #print("Runtime within game loop preparation: ", (time_stop - time_start)/1000000, "ms")

                # Debug code
                # for unit in game_state.units:
                #     trace = "Before sending data\n"
                #     trace = trace + "Game " + str(game) + "\n"
                #     for unit_dbg in game_state.units:
                #         trace = trace + "Unit id " + str(unit_dbg.unit_id) + "\n"
                #         for action_id_dbg in unit_dbg.actions_id:
                #             trace = trace + "Action " + str(action_id_dbg) + "\n"

                if number_units * 2 < total_number_possible_actions:
                    enough_actions = True
                else:
                    enough_actions = False

                if len(game_state.units) > 0 and enough_actions: # if the list is not empty,
                                                                 # and if the number of possible actions is at least greater than twice the number of units 
                    # print("Game ", game)
                    # filename= "game_{:02d}.txt".format(game)
                    # with open(filename, 'w') as f:
                    #     print("Before sending:", file=f)
                    #     for unit in game_state.units:
                    #         print( "Unit ", unit.unit_id, file=f )
                    #         for action in unit.actions_id:
                    #             print( action, file=f )

                    if not sock:
                        #time_start = time.perf_counter_ns()
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(("localhost", 1085))
                        #time_stop = time.perf_counter_ns()
                        #print("Runtime for creating socket: ", (time_stop - time_start)/1000000, "ms")

                    # send the game state to the solver...
                    #time_start = time.perf_counter_ns()
                    sock.send( game_state.SerializeToString() )
                    #time_stop = time.perf_counter_ns()
                    #print("Runtime for sending game state to the C++ server: ", (time_stop - time_start)/1000000, "ms")
                    # ...and wait for its solution
                    solution_from_solver = asr_pb2.State()
                    #time_start = time.perf_counter_ns()
                    solution_from_solver.ParseFromString( sock.recv(65356) ) # Is 65356 sufficient?
                    #time_stop = time.perf_counter_ns()
                    #print("Runtime for receiving solution from the C++ server: ", (time_stop - time_start)/1000000, "ms")

                    # if game == 0:
                    #     print("After receiving:")
                    #     for unit in solution_from_solver.units:
                    #         print( "Unit ", unit.unit_id )
                    #         for action in unit.actions_id:
                    #             print( action )

                    #time_start = time.perf_counter_ns()
                    if solution_from_solver.find_solution:
                        for unit in solution_from_solver.units:
                            # if unit.unit_id < 0 or unit.unit_id >= 256:
                            #     print(trace)
                            #     print("After receiving the results")
                            #     print("Game ", game)
                            #     for unit_dbg in solution_from_solver.units:
                            #         print("Unit id ", unit_dbg.unit_id)
                            #         for action_id_dbg in unit_dbg.actions_id:
                            #             print("Action ", action_id_dbg)
                            # else:
                            #     for action_id in unit.actions_id:
                            #         if action_id < 0 or action_id > 90:
                            #             print(trace)
                            #             print("After receiving the results")
                            #             print("Game ", game)
                            #             for unit_dbg in game_state.units:
                            #                 print("Unit id ", unit_dbg.unit_id)
                            #                 for action_id_dbg in unit_dbg.actions_id:
                            #                     print("Action ", action_id_dbg)

                            for action_id in unit.actions_id:
                                if action_id == 1:
                                    filtered_actions[unit.unit_id][0] = 1.0
                                elif 2 <= action_id <= 5:
                                    filtered_actions[unit.unit_id][1] = 1.0
                                    filtered_actions[unit.unit_id][6 + action_id - 2] = 1.0
                                elif 6 <= action_id <= 9:
                                    filtered_actions[unit.unit_id][2] = 1.0
                                    filtered_actions[unit.unit_id][10 + action_id - 6] = 1.0
                                elif 10 <= action_id <= 13:
                                    filtered_actions[unit.unit_id][3] = 1.0
                                    filtered_actions[unit.unit_id][14 + action_id - 10] = 1.0
                                elif 14 <= action_id <= 41:
                                    filtered_actions[unit.unit_id][4] = 1.0
                                    if action_id <= 20:
                                        filtered_actions[unit.unit_id][18] = 1.0
                                        filtered_actions[unit.unit_id][22 + action_id - 14] = 1.0
                                    elif action_id <= 27:
                                        filtered_actions[unit.unit_id][19] = 1.0
                                        filtered_actions[unit.unit_id][22 + action_id - 21] = 1.0
                                    elif action_id <= 34:
                                        filtered_actions[unit.unit_id][20] = 1.0
                                        filtered_actions[unit.unit_id][22 + action_id - 28] = 1.0
                                    else:
                                        filtered_actions[unit.unit_id][21] = 1.0
                                        filtered_actions[unit.unit_id][22 + action_id - 35] = 1.0
                                else:
                                    filtered_actions[unit.unit_id][5] = 1.0
                                    filtered_actions[unit.unit_id][29 + action_id - 42] = 1.0
                        invalid_action_masks[game] = filtered_actions # Is this legit?
                    else:
                        print("Solution not found")
                    #time_stop = time.perf_counter_ns()
                    #print("Runtime for if solution_from_solver.find_solution: ", (time_stop - time_start)/1000000, "ms")

            # invalid_action_masks [6144, 78], 6144 = 24 games * 256 cells of the grid
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])

            # split_invalid_action_masks ([6144, 6], [6144, 4], [6144, 4], [6144, 4], [6144, 4], [6144, 7], [6144, 49])
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)

            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
            # action [7, 6144] with integers for the 7 chosen parameters of the action
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks, self.critic(hidden)

    def get_value(self, x):
        return self.critic(self.encoder(x))


def run_evaluation(model_path: str, output_path: str, eval_maps: List[str]):
    args = [
        "python",
        "league.py",
        "--evals",
        model_path,
        "--update-db",
        "false",
        "--cuda",
        "false",
        "--output-path",
        output_path,
        "--model-type",
        "ppo_gridnet",
        "--maps",
        *eval_maps,
    ]
    print(f"Evaluating {model_path}")
    with subprocess.Popen(args) as fd:
        return_code = fd.wait()
        assert return_code == 0
    return (model_path, output_path)


class TrueskillWriter:
    def __init__(self, prod_mode, writer, league_path: str, league_step_path: str):
        self.prod_mode = prod_mode
        self.writer = writer
        self.trueskill_df = pd.read_csv(league_path)
        self.trueskill_step_df = pd.read_csv(league_step_path)
        self.trueskill_step_df["type"] = self.trueskill_step_df["name"]
        self.trueskill_step_df["step"] = 0
        # xxx(okachaiev): not sure we need this copy
        self.preset_trueskill_step_df = self.trueskill_step_df.copy()

    def on_evaluation_done(self, future):
        if future.cancelled():
            return
        model_path, output_path = future.result()
        league = pd.read_csv(output_path, index_col="name")
        assert model_path in league.index
        model_global_step = int(model_path.split("/")[-1][:-3])
        self.writer.add_scalar("charts/trueskill", league.loc[model_path]["trueskill"], model_global_step)
        print(f"global_step={model_global_step}, trueskill={league.loc[model_path]['trueskill']}")

        # table visualization logic
        if self.prod_mode:
            trueskill_data = {
                "name": league.loc[model_path].name,
                "mu": league.loc[model_path]["mu"],
                "sigma": league.loc[model_path]["sigma"],
                "trueskill": league.loc[model_path]["trueskill"],
            }
            self.trueskill_df = self.trueskill_df.append(trueskill_data, ignore_index=True)
            wandb.log({"trueskill": wandb.Table(dataframe=self.trueskill_df)})
            trueskill_data["type"] = "training"
            trueskill_data["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(trueskill_data, ignore_index=True)
            preset_trueskill_step_df_clone = self.preset_trueskill_step_df.copy()
            preset_trueskill_step_df_clone["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(preset_trueskill_step_df_clone, ignore_index=True)
            wandb.log({"trueskill_step": wandb.Table(dataframe=self.trueskill_step_df)})



def save_as_csv(f_name, logs):
    #save the logs as a csv file
    (pd.DataFrame.from_dict(data=logs, orient='columns').to_csv(f_name+'.csv', header=True))

if __name__ == "__main__":
    args = parse_args()

    print(f"Save frequency: {args.save_frequency}")

    # Python client to connect to C++ server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 1085))

    # game_state = asr_pb2.State()
    # unit = game_state.units.add()
    # unit.unit_id = 1
    # unit.actions_id.extend([1,3,57])

    # unit = game_state.units.add()
    # unit.unit_id = 3
    # unit.actions_id.extend([6,8,42])

    # sock.send( game_state.SerializeToString() )
    # game_state.ParseFromString( sock.recv(1024) )
    # for unit in game_state.units:
    #     print( "Unit ", unit.unit_id )
    #     for action in unit.actions_id:
    #         print( action )


    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
        + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        lr = lambda f: f * args.learning_rate

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    action_space_shape = (mapsize, len(envs.action_plane_space.nvec))
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # CRASH AND RESUME LOGIC:
    starting_update = 1

    if args.prod_mode and wandb.run.resumed:
        starting_update = run.summary.get("charts/update") + 1
        global_step = starting_update * args.batch_size
        api = wandb.Api()
        run = api.run(f"{run.entity}/{run.project}/{run.id}")
        model = run.file("agent.pt")
        model.download(f"models/{experiment_name}/")
        agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
        agent.eval()
        print(f"resumed at update {starting_update}")

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum(param.nelement() for param in agent.parameters())
    print("Model's total parameters:", total_params)

    # EVALUATION LOGIC:
    trueskill_writer = TrueskillWriter(
        args.prod_mode, writer, "gym-microrts-static-files/league.csv", "gym-microrts-static-files/league.csv"
    )

    logs = defaultdict(list)

    for update in range(starting_update, args.num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            # envs.render()
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks[step] = torch.tensor(envs.get_action_mask()).to(device)
                # Assign actions to each unit here
                #time_start = time.perf_counter_ns()
                action, logproba, _, _, vs = agent.get_action_and_value(
                    next_obs, envs=envs, num_envs=args.num_envs, invalid_action_masks=invalid_action_masks[step], device=device, sock=sock
                )
                #time_stop = time.perf_counter_ns()
                #print("Runtime for calling get action (no grad): ", (time_stop - time_start)/1000000, "ms")
                # raise "Kill"
                values[step] = vs.flatten()

            actions[step] = action
            logprobs[step] = logproba
            try:
                next_obs, rs, ds, infos = envs.step(action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise
            rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    for key in info["microrts_stats"]:
                        writer.add_scalar(f"charts/episodic_return/{key}", info["microrts_stats"][key], global_step)

                    logs["steps"].append(global_step)
                    logs["episodic_return"].append(float(info['episode']['r']))
                    logs["episodic_length"].append(float(info["episode"]["l"]))
                    for key in info["microrts_stats"]:
                        logs[key].append(info["microrts_stats"][key])
                    save_as_csv("runs/"+experiment_name,logs)
                    break

        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

        # Optimizing the policy and value network
        inds = np.arange(
            args.batch_size,
        )
        for i_epoch_pi in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                #time_start = time.perf_counter_ns()
                _, newlogproba, entropy, _, new_values = agent.get_action_and_value(
                    b_obs[minibatch_ind], b_actions.long()[minibatch_ind], b_invalid_action_masks[minibatch_ind], envs, args.num_envs, device, sock
                )
                #time_stop = time.perf_counter_ns()
                #print("Runtime for calling get action (i_epoch ", i_epoch_pi, ", start ", start, "): ", (time_stop - time_start)/1000000, "ms")
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                    v_clipped = b_values[minibatch_ind] + torch.clamp(
                        new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if (update - 1) % args.save_frequency == 0:
            if not os.path.exists(f"models/{experiment_name}"):
                os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")
            torch.save(agent.state_dict(), f"models/{experiment_name}/{global_step}.pt")
            if args.prod_mode:
                wandb.save(f"models/{experiment_name}/agent.pt", base_path=f"models/{experiment_name}", policy="now")
            if eval_executor is not None:
                future = eval_executor.submit(
                    run_evaluation,
                    f"models/{experiment_name}/{global_step}.pt",
                    f"runs/{experiment_name}/{global_step}.csv",
                    args.eval_maps,
                )
                print(f"Queued models/{experiment_name}/{global_step}.pt")
                future.add_done_callback(trueskill_writer.on_evaluation_done)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/update", update, global_step)
        writer.add_scalar("losses/value_loss", v_loss.detach().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.detach().item(), global_step)
        writer.add_scalar("losses/entropy", entropy.detach().mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.detach().item(), global_step)
        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    if eval_executor is not None:
        # shutdown pool of threads but make sure we finished scheduled evaluations
        eval_executor.shutdown(wait=True, cancel_futures=False)
    envs.close()
    writer.close()
