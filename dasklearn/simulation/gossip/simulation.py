import math

import networkx as nx

from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.gossip.client import GossipClient
from dasklearn.simulation.events import *
from dasklearn.simulation.simulation import Simulation


class GossipSimulation(Simulation):
    CLIENT_CLASS = GossipClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)

        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(DISSEMINATE, "disseminate")
        self.register_event_callback(TEST, "test")
