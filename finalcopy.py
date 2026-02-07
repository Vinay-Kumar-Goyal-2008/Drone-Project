import mediapipe as mp
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from time import time
import threading
import numpy as np
import pickle

framespersample=40
votinglength=11
minvotes=6

