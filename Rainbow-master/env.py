# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch


class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)#创建一个 2 x 84 x 84 的零张量用于缓存最后两个原始预处理帧
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))#积累reward
      #在 t==2 和 t==3 时调用 self._get_state() 将环境当前帧（通常是经过灰度、裁剪、缩放到 84x84 的预处理图像）
      # 保存到 frame_buffer 的两个槽里。只取最后两帧用于后续的 max-pooling
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]#在 t==2 和 t==3 时调用 self._get_state() 将环境当前帧（通常是经过灰度、裁剪、缩放到 84x84 的预处理图像）
    #保存到 frame_buffer 的两个槽里。只取最后两帧用于后续的 max-pooling
    #沿第 0 维（即两个帧的维度）做逐像素最大值，返回 (values, indices)，取 [0] 得到 values。结果是一个 84 x 84 的张量，代表本次 step 的“观测帧”
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:#训练模式下，失去一条命当成一次终止
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done#把 state_buffer 中的若干帧沿第 0 维堆叠起来，返回形状为 (history_length, 84, 84) 的张量，作为网络的输入状态

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
