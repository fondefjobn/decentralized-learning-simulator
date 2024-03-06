from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task

import random


class GossipClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        # Not available by default
        self.available = False
        self.initialization = True
        self.finished = False

        random.seed(self.simulator.settings.seed)

    def init_client(self, event: Event):
        self.round = 1
        # Initialize the model
        start_train_event = Event(event.time, self.index, START_TRAIN)
        self.simulator.schedule(start_train_event)

    def finish_train(self, event: Event):
        # Update the model after the training was finished
        self.own_model = event.data["model"]
        self.available = True
        self.round += 1

        if self.round >= self.simulator.settings.rounds:
            self.finished = True

        if self.initialization:
            # Schedule a disseminate action
            disseminate_event = Event(event.time, self.index, DISSEMINATE)
            self.simulator.schedule(disseminate_event)
            self.initialization = False

    def on_incoming_model(self, event: Event):
        # We received a model
        self.client_log("Client %d received from %d model %s in round %d" % (self.index, event.data["from"], event.data["model"], self.round))

        # Check if available
        if self.available and self.round <= self.simulator.settings.rounds:
            # Lock
            self.available = False

            # Aggregate the incoming and own models
            model_names = [event.data["model"], self.own_model]
            self.client_log("Client %d will aggregate in round %d (%s)" % (self.index, self.round, model_names))
            self.own_model = self.aggregate_models(model_names)

            # Should we test?
            if self.round % self.simulator.settings.test_interval == 0:
                test_task_name = "test_%d_%d" % (self.index, self.round)
                task = Task(test_task_name, "test", data={"model": self.own_model, "round": self.round, "time": self.simulator.current_time, "peer": self.index})
                self.add_compute_task(task)
                self.own_model = test_task_name

            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)

    def disseminate(self, event: Event):
        # Send the model to a random peer
        peers = [peer for peer in range(self.simulator.settings.participants)]
        peers.remove(self.index)
        sample_peers = random.choice(peers)
        for peer in sample_peers:
            self.send_model(peer, self.own_model)

        # Schedule next disseminate action if not all clients has finished
        all_finished = all(map(lambda client: client.finished, self.simulator.clients))
        if not all_finished:
            # TODO: replace 10 with argument
            disseminate_event = Event(event.time + 10, self.index, DISSEMINATE)
            self.simulator.schedule(disseminate_event)
