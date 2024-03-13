from dasklearn.simulation.client import BaseClient
from dasklearn.simulation.events import *
from dasklearn.tasks.task import Task


class GossipClient(BaseClient):

    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)

        # Not available by default
        self.available = False

    def init_client(self, event: Event):
        # Initialize the model
        self.round = 0
        # Schedule train, test and disseminate events
        start_train_event = Event(event.time, self.index, START_TRAIN)
        self.simulator.schedule(start_train_event)
        if event.time + self.simulator.settings.period <= self.simulator.settings.duration:
            disseminate_event = Event(event.time + self.simulator.settings.period, self.index, DISSEMINATE)
            self.simulator.schedule(disseminate_event)
        if event.time + self.simulator.settings.test_period <= self.simulator.settings.duration:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)

    def finish_train(self, event: Event):
        # Update the model after the training was finished
        self.own_model = event.data["model"]
        self.available = True
        self.round += 1

    def on_incoming_model(self, event: Event):
        # We received a model
        self.client_log("Client %d received from %d model %s" % (self.index, event.data["from"], event.data["model"]))

        # Check if available
        if self.available and event.time <= self.simulator.settings.duration:
            # Lock
            self.available = False

            # Aggregate the incoming and own models
            model_names = [event.data["model"], self.own_model]
            # compute weights
            rounds = [event.data["metadata"]["rounds"], self.round]
            weights = list(map(lambda x: x / sum(rounds), rounds))
            self.round = max(rounds)
            self.client_log("Client %d will aggregate and train (%s)" % (self.index, model_names))
            self.own_model = self.aggregate_models(model_names, weights)

            # Schedule a train action
            start_train_event = Event(event.time, self.index, START_TRAIN)
            self.simulator.schedule(start_train_event)

    def disseminate(self, event: Event):
        # Send the model to a random peer other than oneself
        if self.round > 0:
            peer = self.simulator.get_random_participant(self.index)
            metadata = dict(rounds=self.round)
            self.send_model(peer, self.own_model, metadata)

            self.client_log("Client %d will send model %s to %d" % (self.index, self.own_model, peer))

        # Schedule next disseminate action
        if event.time + self.simulator.settings.period <= self.simulator.settings.duration:
            disseminate_event = Event(event.time + self.simulator.settings.period, self.index, DISSEMINATE)
            self.simulator.schedule(disseminate_event)

    def test(self, event: Event):
        if self.round > 0:
            self.client_log("Client %d will test its model %s" % (self.index, self.own_model))

            test_task_name = "test_%d_%d" % (self.index, event.time)
            task = Task(test_task_name, "test", data={"model": self.own_model, "time": self.simulator.current_time, "peer": self.index, "rounds":self.round})
            self.add_compute_task(task)
            self.own_model = test_task_name

        # Schedule next test action
        if event.time + self.simulator.settings.test_period <= self.simulator.settings.duration:
            test_event = Event(event.time + self.simulator.settings.test_period, self.index, TEST)
            self.simulator.schedule(test_event)
